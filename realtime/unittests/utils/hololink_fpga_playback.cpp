/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_fpga_playback.cpp
/// @brief Generic RPC playback tool for Hololink FPGA / emulator testing.
///
/// Sends RPC messages to the FPGA (or emulator) via the Hololink UDP control
/// plane, triggering RDMA transmission to the bridge.  After playback, reads
/// back responses from the ILA capture RAM and verifies them.
///
/// For the generic bridge, the payload is a sequence of ascending bytes and
/// the expected response is each byte incremented by 1.
///
/// Usage:
///   ./hololink_fpga_playback \
///       --control-ip=10.0.0.2 --control-port=8193 \
///       --bridge-qp=0x5 --bridge-rkey=12345 --bridge-buffer=0x7f... \
///       --page-size=384 --num-pages=64 --num-shots=100

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

//==============================================================================
// Hololink Control Plane Protocol
//==============================================================================

static constexpr uint8_t WR_DWORD = 0x04;
static constexpr uint8_t WR_BLOCK = 0x09;
static constexpr uint8_t RD_DWORD = 0x14;
static constexpr uint8_t RD_BLOCK = 0x19;
static constexpr uint8_t REQUEST_FLAGS_ACK_REQUEST = 0x01;
static constexpr uint8_t RESPONSE_SUCCESS = 0x00;

// VP register offsets
static constexpr uint32_t DP_QP = 0x00;
static constexpr uint32_t DP_RKEY = 0x04;
static constexpr uint32_t DP_PAGE_LSB = 0x08;
static constexpr uint32_t DP_PAGE_MSB = 0x0C;
static constexpr uint32_t DP_PAGE_INC = 0x10;
static constexpr uint32_t DP_MAX_BUFF = 0x14;
static constexpr uint32_t DP_BUFFER_LENGTH = 0x18;
static constexpr uint32_t DP_HOST_IP = 0x28;

// HIF register offsets
static constexpr uint32_t DP_VP_MASK = 0x0C;

// Player registers
static constexpr uint32_t PLAYER_BASE = 0x50000000;
static constexpr uint32_t PLAYER_ENABLE = PLAYER_BASE + 0x04;
static constexpr uint32_t PLAYER_TIMER = PLAYER_BASE + 0x08;
static constexpr uint32_t PLAYER_WIN_SIZE = PLAYER_BASE + 0x0C;
static constexpr uint32_t PLAYER_WIN_NUM = PLAYER_BASE + 0x10;

// Playback BRAM
static constexpr uint32_t RAM_BASE = 0x50100000;
static constexpr int BRAM_NUM_BANKS = 16;
static constexpr int BRAM_W_SAMPLE_ADDR = 9;
static constexpr int BRAM_BANK_STRIDE = 1 << (BRAM_W_SAMPLE_ADDR + 2);

// ILA capture
static constexpr uint32_t ILA_BASE = 0x40000000;
static constexpr uint32_t ILA_CTRL = ILA_BASE + 0x00;
static constexpr uint32_t ILA_STATUS = ILA_BASE + 0x80;
static constexpr uint32_t ILA_SAMPLE_ADDR = ILA_BASE + 0x84;
static constexpr uint32_t ILA_DATA_BASE = 0x40100000;
static constexpr int ILA_NUM_BANKS = 17;
static constexpr int ILA_W_ADDR = 13;
static constexpr int ILA_BANK_STRIDE = 1 << (ILA_W_ADDR + 2);

// Hololink page encoding
static constexpr int PAGE_SHIFT = 7; // 128-byte pages

//==============================================================================
// UDP helpers
//==============================================================================

static void write_be32(uint8_t *p, uint32_t v) {
  p[0] = (v >> 24) & 0xFF;
  p[1] = (v >> 16) & 0xFF;
  p[2] = (v >> 8) & 0xFF;
  p[3] = v & 0xFF;
}

static void write_be16(uint8_t *p, uint16_t v) {
  p[0] = (v >> 8) & 0xFF;
  p[1] = v & 0xFF;
}

static uint32_t read_be32(const uint8_t *p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | p[3];
}

//==============================================================================
// Control plane client
//==============================================================================

class ControlPlaneClient {
public:
  bool connect(const std::string &ip, uint16_t port) {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0)
      return false;

    addr_.sin_family = AF_INET;
    addr_.sin_port = htons(port);
    inet_pton(AF_INET, ip.c_str(), &addr_.sin_addr);

    // Set receive timeout
    timeval tv{2, 0};
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    return true;
  }

  ~ControlPlaneClient() {
    if (fd_ >= 0)
      ::close(fd_);
  }

  bool write_dword(uint32_t addr, uint32_t value) {
    uint8_t pkt[14];
    pkt[0] = WR_DWORD;
    pkt[1] = REQUEST_FLAGS_ACK_REQUEST;
    write_be16(pkt + 2, seq_++);
    pkt[4] = 0;
    pkt[5] = 0;
    write_be32(pkt + 6, addr);
    write_be32(pkt + 10, value);

    sendto(fd_, pkt, sizeof(pkt), 0, reinterpret_cast<sockaddr *>(&addr_),
           sizeof(addr_));

    // Wait for ACK
    uint8_t resp[16];
    ssize_t n = recv(fd_, resp, sizeof(resp), 0);
    return (n >= 5 && resp[4] == RESPONSE_SUCCESS);
  }

  bool write_block(const std::vector<std::pair<uint32_t, uint32_t>> &pairs) {
    std::vector<uint8_t> pkt(6 + pairs.size() * 8);
    pkt[0] = WR_BLOCK;
    pkt[1] = REQUEST_FLAGS_ACK_REQUEST;
    write_be16(pkt.data() + 2, seq_++);
    pkt[4] = 0;
    pkt[5] = 0;

    size_t off = 6;
    for (auto &[addr, val] : pairs) {
      write_be32(pkt.data() + off, addr);
      write_be32(pkt.data() + off + 4, val);
      off += 8;
    }

    sendto(fd_, pkt.data(), pkt.size(), 0, reinterpret_cast<sockaddr *>(&addr_),
           sizeof(addr_));

    uint8_t resp[16];
    ssize_t n = recv(fd_, resp, sizeof(resp), 0);
    return (n >= 5 && resp[4] == RESPONSE_SUCCESS);
  }

  uint32_t read_dword(uint32_t addr) {
    uint8_t pkt[10];
    pkt[0] = RD_DWORD;
    pkt[1] = REQUEST_FLAGS_ACK_REQUEST;
    write_be16(pkt + 2, seq_++);
    pkt[4] = 0;
    pkt[5] = 0;
    write_be32(pkt + 6, addr);

    sendto(fd_, pkt, sizeof(pkt), 0, reinterpret_cast<sockaddr *>(&addr_),
           sizeof(addr_));

    uint8_t resp[32];
    ssize_t n = recv(fd_, resp, sizeof(resp), 0);
    if (n >= 14)
      return read_be32(resp + 10);
    return 0;
  }

private:
  int fd_ = -1;
  sockaddr_in addr_{};
  uint16_t seq_ = 0;
};

//==============================================================================
// Arguments
//==============================================================================

struct PlaybackArgs {
  std::string control_ip = "10.0.0.2";
  uint16_t control_port = 8193;
  uint32_t bridge_qp = 0;
  uint32_t bridge_rkey = 0;
  uint64_t bridge_buffer = 0;
  size_t page_size = 384;
  unsigned num_pages = 64;
  uint32_t num_shots = 100;
  uint32_t payload_size = 8; // bytes of RPC argument data
  uint32_t vp_address = 0x1000;
  uint32_t hif_address = 0x0800;
  std::string bridge_ip = "10.0.0.1";
  bool verify = true;
};

static PlaybackArgs parse_args(int argc, char *argv[]) {
  PlaybackArgs args;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a.find("--control-ip=") == 0)
      args.control_ip = a.substr(13);
    else if (a.find("--control-port=") == 0)
      args.control_port = std::stoi(a.substr(15));
    else if (a.find("--bridge-qp=") == 0)
      args.bridge_qp = std::stoul(a.substr(12), nullptr, 0);
    else if (a.find("--bridge-rkey=") == 0)
      args.bridge_rkey = std::stoul(a.substr(14), nullptr, 0);
    else if (a.find("--bridge-buffer=") == 0)
      args.bridge_buffer = std::stoull(a.substr(16), nullptr, 0);
    else if (a.find("--page-size=") == 0)
      args.page_size = std::stoull(a.substr(12));
    else if (a.find("--num-pages=") == 0)
      args.num_pages = std::stoul(a.substr(12));
    else if (a.find("--num-shots=") == 0)
      args.num_shots = std::stoul(a.substr(12));
    else if (a.find("--payload-size=") == 0)
      args.payload_size = std::stoul(a.substr(15));
    else if (a.find("--vp-address=") == 0)
      args.vp_address = std::stoul(a.substr(13), nullptr, 0);
    else if (a.find("--hif-address=") == 0)
      args.hif_address = std::stoul(a.substr(14), nullptr, 0);
    else if (a.find("--bridge-ip=") == 0)
      args.bridge_ip = a.substr(12);
    else if (a == "--no-verify")
      args.verify = false;
    else if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: hololink_fpga_playback [options]\n"
          << "\nGeneric RPC playback tool for Hololink FPGA/emulator.\n"
          << "\nOptions:\n"
          << "  --control-ip=ADDR     Emulator/FPGA IP (default: 10.0.0.2)\n"
          << "  --control-port=N      UDP control port (default: 8193)\n"
          << "  --bridge-qp=N         Bridge QP number\n"
          << "  --bridge-rkey=N       Bridge RKey\n"
          << "  --bridge-buffer=ADDR  Bridge buffer address\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages=N         Number of ring buffer slots (default: "
             "64)\n"
          << "  --num-shots=N         Number of RPC messages (default: 100)\n"
          << "  --payload-size=N      Bytes per RPC payload (default: 8)\n"
          << "  --vp-address=ADDR     VP register base (default: 0x1000)\n"
          << "  --hif-address=ADDR    HIF register base (default: 0x0800)\n"
          << "  --bridge-ip=ADDR      Bridge IP for FPGA (default: 10.0.0.1)\n"
          << "  --no-verify           Skip ILA correction verification\n";
      exit(0);
    }
  }
  return args;
}

//==============================================================================
// BRAM loading
//==============================================================================

/// Build one RPC message for the increment handler.
/// Format: RPCHeader + ascending byte payload.
static std::vector<uint8_t> build_rpc_message(uint32_t shot_index,
                                              uint32_t payload_size) {
  using cudaq::nvqlink::fnv1a_hash;
  using cudaq::nvqlink::RPCHeader;

  constexpr uint32_t FUNC_ID = fnv1a_hash("rpc_increment");

  std::vector<uint8_t> msg(sizeof(RPCHeader) + payload_size, 0);
  auto *hdr = reinterpret_cast<RPCHeader *>(msg.data());
  hdr->magic = cudaq::nvqlink::RPC_MAGIC_REQUEST;
  hdr->function_id = FUNC_ID;
  hdr->arg_len = payload_size;

  uint8_t *payload = msg.data() + sizeof(RPCHeader);
  for (uint32_t i = 0; i < payload_size; i++) {
    payload[i] = static_cast<uint8_t>((shot_index + i) & 0xFF);
  }
  return msg;
}

/// Spread a message across 16 BRAM banks (64-byte beats).
static void load_message_to_bram(ControlPlaneClient &ctrl,
                                 const std::vector<uint8_t> &msg,
                                 uint32_t window_index,
                                 uint32_t cycles_per_window) {
  std::vector<std::pair<uint32_t, uint32_t>> batch;

  for (uint32_t cycle = 0; cycle < cycles_per_window; cycle++) {
    uint32_t sample = window_index * cycles_per_window + cycle;
    for (int bank = 0; bank < BRAM_NUM_BANKS; bank++) {
      uint32_t addr =
          RAM_BASE + (bank << (BRAM_W_SAMPLE_ADDR + 2)) + (sample * 4);
      uint32_t val = 0;
      size_t byte_off = cycle * 64 + bank * 4;
      if (byte_off < msg.size()) {
        size_t copy_len = std::min<size_t>(4, msg.size() - byte_off);
        memcpy(&val, msg.data() + byte_off, copy_len);
      }
      batch.push_back({addr, val});
    }

    // Send in chunks to stay within UDP MTU
    if (batch.size() >= 64) {
      ctrl.write_block(batch);
      batch.clear();
    }
  }

  if (!batch.empty())
    ctrl.write_block(batch);
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char *argv[]) {
  auto args = parse_args(argc, argv);

  std::cout << "=== Hololink Generic RPC Playback ===" << std::endl;
  std::cout << "Control: " << args.control_ip << ":" << args.control_port
            << std::endl;
  std::cout << "Shots: " << args.num_shots << std::endl;
  std::cout << "Payload size: " << args.payload_size << " bytes" << std::endl;

  ControlPlaneClient ctrl;
  if (!ctrl.connect(args.control_ip, args.control_port)) {
    std::cerr << "ERROR: Failed to connect to control plane" << std::endl;
    return 1;
  }

  //============================================================================
  // Configure RDMA target (bridge's QP/RKEY/buffer)
  //============================================================================
  std::cout << "\n[1/4] Configuring RDMA target..." << std::endl;

  uint32_t vp = args.vp_address;
  ctrl.write_dword(vp + DP_QP, args.bridge_qp);
  ctrl.write_dword(vp + DP_RKEY, args.bridge_rkey);
  ctrl.write_dword(vp + DP_PAGE_LSB,
                   static_cast<uint32_t>(args.bridge_buffer >> PAGE_SHIFT));
  ctrl.write_dword(vp + DP_PAGE_MSB,
                   static_cast<uint32_t>(args.bridge_buffer >> 32));
  ctrl.write_dword(vp + DP_PAGE_INC,
                   static_cast<uint32_t>(args.page_size >> PAGE_SHIFT));
  ctrl.write_dword(vp + DP_MAX_BUFF, args.num_pages - 1);

  size_t frame_size = sizeof(cudaq::nvqlink::RPCHeader) + args.payload_size;
  ctrl.write_dword(vp + DP_BUFFER_LENGTH, static_cast<uint32_t>(frame_size));

  // Set bridge IP for emulator GID derivation
  {
    in_addr a;
    inet_pton(AF_INET, args.bridge_ip.c_str(), &a);
    ctrl.write_dword(vp + DP_HOST_IP, a.s_addr);
  }

  // Enable VP mask
  ctrl.write_dword(args.hif_address + DP_VP_MASK, 0x01);

  std::cout << "  Bridge QP: 0x" << std::hex << args.bridge_qp << std::dec
            << std::endl;
  std::cout << "  Bridge RKey: " << args.bridge_rkey << std::endl;
  std::cout << "  Bridge Buffer: 0x" << std::hex << args.bridge_buffer
            << std::dec << std::endl;

  //============================================================================
  // Load RPC messages into BRAM
  //============================================================================
  std::cout << "\n[2/4] Loading RPC messages into BRAM..." << std::endl;

  uint32_t window_size = static_cast<uint32_t>(frame_size);
  uint32_t cycles_per_window = (window_size + 63) / 64;

  for (uint32_t shot = 0; shot < args.num_shots; shot++) {
    auto msg = build_rpc_message(shot, args.payload_size);
    load_message_to_bram(ctrl, msg, shot, cycles_per_window);

    if ((shot + 1) % 10 == 0)
      std::cout << "  Loaded " << (shot + 1) << "/" << args.num_shots
                << std::endl;
  }

  //============================================================================
  // Arm ILA and trigger playback
  //============================================================================
  std::cout << "\n[3/4] Triggering playback..." << std::endl;

  // Arm ILA capture
  if (args.verify) {
    ctrl.write_dword(ILA_CTRL, 0x01);
  }

  // Set player registers
  ctrl.write_dword(PLAYER_WIN_SIZE, window_size);
  ctrl.write_dword(PLAYER_WIN_NUM, args.num_shots);
  ctrl.write_dword(PLAYER_TIMER, 322 * 100); // 100 us spacing

  // Trigger
  ctrl.write_dword(PLAYER_ENABLE, 1);
  std::cout << "  Playback triggered for " << args.num_shots << " shots"
            << std::endl;

  //============================================================================
  // Wait and verify ILA capture
  //============================================================================
  if (args.verify) {
    std::cout << "\n[4/4] Verifying responses..." << std::endl;

    // Wait for ILA to indicate done (bit 1 of ILA_STATUS)
    int timeout = 120; // seconds
    bool done = false;
    for (int i = 0; i < timeout * 10 && !done; i++) {
      uint32_t status = ctrl.read_dword(ILA_STATUS);
      if (status & 0x02)
        done = true;
      else
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!done) {
      std::cerr << "ERROR: ILA capture timeout" << std::endl;
      return 1;
    }

    uint32_t sample_count = ctrl.read_dword(ILA_SAMPLE_ADDR);
    std::cout << "  ILA captured " << sample_count << " samples" << std::endl;

    // Read back and verify each response
    uint32_t matched = 0;
    uint32_t check_count = std::min(sample_count, args.num_shots);

    for (uint32_t i = 0; i < check_count; i++) {
      // Read response from ILA banks (the first bytes are RPCResponse header)
      std::vector<uint8_t> response_bytes(64, 0);
      for (int bank = 0; bank < std::min(ILA_NUM_BANKS - 1, 16); bank++) {
        uint32_t addr = ILA_DATA_BASE + (bank << (ILA_W_ADDR + 2)) + (i * 4);
        uint32_t val = ctrl.read_dword(addr);
        size_t byte_off = bank * 4;
        if (byte_off + 4 <= response_bytes.size())
          memcpy(response_bytes.data() + byte_off, &val, 4);
      }

      // Check control signals (bank 16): tvalid must be set
      uint32_t ctrl_addr =
          ILA_DATA_BASE + ((ILA_NUM_BANKS - 1) << (ILA_W_ADDR + 2)) + (i * 4);
      uint32_t ctrl_val = ctrl.read_dword(ctrl_addr);
      bool tvalid = (ctrl_val & 0x01) != 0;

      if (!tvalid) {
        std::cerr << "  Shot " << i << ": tvalid=0 (no response)" << std::endl;
        continue;
      }

      // Parse RPCResponse
      auto *resp = reinterpret_cast<const cudaq::nvqlink::RPCResponse *>(
          response_bytes.data());

      if (resp->magic != cudaq::nvqlink::RPC_MAGIC_RESPONSE) {
        std::cerr << "  Shot " << i << ": bad magic 0x" << std::hex
                  << resp->magic << std::dec << std::endl;
        continue;
      }

      if (resp->status != 0) {
        std::cerr << "  Shot " << i << ": error status " << resp->status
                  << std::endl;
        continue;
      }

      // Verify increment: each byte should be (shot_index + byte_index + 1)
      const uint8_t *result_data =
          response_bytes.data() + sizeof(cudaq::nvqlink::RPCResponse);
      bool ok = true;
      uint32_t check_len = std::min(resp->result_len, args.payload_size);
      for (uint32_t j = 0; j < check_len && ok; j++) {
        uint8_t expected = static_cast<uint8_t>(((i + j) & 0xFF) + 1);
        if (result_data[j] != expected) {
          std::cerr << "  Shot " << i << " byte " << j << ": expected "
                    << (int)expected << " got " << (int)result_data[j]
                    << std::endl;
          ok = false;
        }
      }
      if (ok)
        matched++;
    }

    std::cout << "\n=== Verification Results ===" << std::endl;
    std::cout << "  RPC responses matched: " << matched << " / " << check_count
              << std::endl;

    if (matched == check_count) {
      std::cout << "\n*** ALL RESPONSES VERIFIED ***" << std::endl;
      return 0;
    } else {
      std::cout << "\n*** VERIFICATION FAILED ***" << std::endl;
      return 1;
    }
  } else {
    std::cout << "\n[4/4] Verification skipped (--no-verify)" << std::endl;
    // Wait a bit for playback to complete
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "\n*** PLAYBACK COMPLETE ***" << std::endl;
    return 0;
  }
}
