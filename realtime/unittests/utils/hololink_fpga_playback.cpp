/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_fpga_playback.cpp
/// @brief Generic RPC playback tool for Hololink FPGA / emulator testing.
///
/// Sends RPC messages to the FPGA (or emulator) via the Hololink control
/// plane, triggering RDMA transmission to the bridge.  After playback, reads
/// back responses from the ILA capture RAM and verifies them.
///
/// For the generic bridge, the payload is a sequence of ascending bytes and
/// the expected response is each byte incremented by 1.
///
/// Usage:
///   ./hololink_fpga_playback \
///       --hololink 192.168.0.2 \
///       --bridge-qp=0x5 --bridge-rkey=12345 --bridge-buffer=0x7f... \
///       --page-size=384 --num-pages=128 --num-messages=100

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include <hololink/core/data_channel.hpp>
#include <hololink/core/enumerator.hpp>
#include <hololink/core/hololink.hpp>
#include <hololink/core/metadata.hpp>
#include <hololink/core/timeout.hpp>

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

namespace {

// ============================================================================
// Playback BRAM Constants
// ============================================================================
constexpr std::uint32_t PLAYER_ADDR = 0x5000'0000;
constexpr std::uint32_t RAM_ADDR = 0x5010'0000;
constexpr std::uint32_t PLAYER_TIMER_OFFSET = 0x0008;
constexpr std::uint32_t PLAYER_WINDOW_SIZE_OFFSET = 0x000C;
constexpr std::uint32_t PLAYER_WINDOW_NUMBER_OFFSET = 0x0010;
constexpr std::uint32_t PLAYER_ENABLE_OFFSET = 0x0004;
constexpr std::uint32_t RAM_NUM = 16;
constexpr std::uint32_t RAM_DEPTH = 512;

constexpr std::uint32_t PLAYER_ENABLE_SINGLEPASS = 0x0000'000D;
constexpr std::uint32_t PLAYER_DISABLE = 0x0000'0000;

constexpr std::uint32_t SIF_TX_THRESHOLD_ADDR = 0x0120'0000;
constexpr std::uint32_t SIF_TX_THRESHOLD_IMMEDIATE = 0x0000'0005;

constexpr std::uint32_t METADATA_PACKET_ADDR = 0x102C;

constexpr std::uint32_t DEFAULT_TIMER_SPACING_US = 120;
constexpr std::uint32_t RF_SOC_TIMER_SCALE = 322;

// ============================================================================
// ILA Capture Constants (SIF TX at 0x4000_0000)
//
// Each captured sample is 585 bits:
//   [511:0]   sif_tx_axis_tdata_0
//   [512]     sif_tx_axis_tvalid_0
//   [513]     sif_tx_axis_tlast_0
//   [520:514] sif_ila_wr_tcnt_0
//   [584:521] current_ptp_timestamp {sec[31:0], nsec[31:0]}
// ============================================================================
constexpr std::uint32_t ILA_BASE_ADDR = 0x4000'0000;
constexpr std::uint32_t ILA_CTRL_OFFSET = 0x0000;
constexpr std::uint32_t ILA_SAMPLE_ADDR_OFFSET = 0x0084;
constexpr std::uint32_t ILA_W_DATA = 585;
constexpr std::uint32_t ILA_NUM_RAM = (ILA_W_DATA + 31) / 32; // 19
constexpr std::uint32_t ILA_W_ADDR = 13;                      // log2(8192)
constexpr std::uint32_t ILA_W_RAM = 5;                        // ceil(log2(19))

constexpr std::uint32_t ILA_CTRL_ENABLE = 0x0000'0001;
constexpr std::uint32_t ILA_CTRL_RESET = 0x0000'0002;
constexpr std::uint32_t ILA_CTRL_DISABLE = 0x0000'0000;

constexpr std::uint32_t ILA_TVALID_BIT = 512;

constexpr std::uint32_t ROCEV2_UDP_PORT = 4791;

// ============================================================================
// Arguments
// ============================================================================

struct PlaybackArgs {
  std::string hololink_ip = "192.168.0.2";
  uint16_t control_port = 8192;
  uint32_t bridge_qp = 0;
  uint32_t bridge_rkey = 0;
  uint64_t bridge_buffer = 0;
  size_t page_size = 384;
  unsigned num_pages = 128;
  uint32_t num_messages = 100;
  uint32_t payload_size = 8;
  uint32_t vp_address = 0x1000;
  uint32_t hif_address = 0x0800;
  std::string bridge_ip = "10.0.0.1";
  bool verify = true;
  bool emulator = false;
  bool forward = false; ///< Forward (echo) mode: accept RPC_MAGIC_REQUEST
};

PlaybackArgs parse_args(int argc, char *argv[]) {
  PlaybackArgs args;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 < argc)
        return argv[++i];
      return {};
    };
    auto val_of = [&](const std::string &prefix) -> std::string {
      return a.substr(prefix.size());
    };
    if (a.find("--hololink=") == 0)
      args.hololink_ip = val_of("--hololink=");
    else if (a == "--hololink")
      args.hololink_ip = next();
    else if (a.find("--control-ip=") == 0)
      args.hololink_ip = val_of("--control-ip=");
    else if (a.find("--control-port=") == 0)
      args.control_port = std::stoi(val_of("--control-port="));
    else if (a.find("--bridge-qp=") == 0)
      args.bridge_qp = std::stoul(val_of("--bridge-qp="), nullptr, 0);
    else if (a.find("--bridge-rkey=") == 0)
      args.bridge_rkey = std::stoul(val_of("--bridge-rkey="), nullptr, 0);
    else if (a.find("--bridge-buffer=") == 0)
      args.bridge_buffer = std::stoull(val_of("--bridge-buffer="), nullptr, 0);
    else if (a.find("--page-size=") == 0)
      args.page_size = std::stoull(val_of("--page-size="));
    else if (a.find("--num-pages=") == 0)
      args.num_pages = std::stoul(val_of("--num-pages="));
    else if (a == "--num-pages")
      args.num_pages = std::stoul(next());
    else if (a.find("--num-messages=") == 0)
      args.num_messages = std::stoul(val_of("--num-messages="));
    else if (a == "--num-messages")
      args.num_messages = std::stoul(next());
    else if (a.find("--payload-size=") == 0)
      args.payload_size = std::stoul(val_of("--payload-size="));
    else if (a.find("--vp-address=") == 0)
      args.vp_address = std::stoul(val_of("--vp-address="), nullptr, 0);
    else if (a.find("--hif-address=") == 0)
      args.hif_address = std::stoul(val_of("--hif-address="), nullptr, 0);
    else if (a.find("--bridge-ip=") == 0)
      args.bridge_ip = val_of("--bridge-ip=");
    else if (a == "--no-verify")
      args.verify = false;
    else if (a == "--emulator")
      args.emulator = true;
    else if (a == "--forward")
      args.forward = true;
    else if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: hololink_fpga_playback [options]\n"
          << "\nGeneric RPC playback tool for Hololink FPGA/emulator.\n"
          << "\nOptions:\n"
          << "  --hololink ADDR       FPGA/emulator IP (default: 192.168.0.2)\n"
          << "  --control-port=N      UDP control port (default: 8192)\n"
          << "  --bridge-qp=N         Bridge QP number\n"
          << "  --bridge-rkey=N       Bridge RKey\n"
          << "  --bridge-buffer=ADDR  Bridge buffer address\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages N         Ring buffer slots (default: 128)\n"
          << "  --num-messages N      Number of RPC messages (default: 100)\n"
          << "  --payload-size=N      Bytes per RPC payload (default: 8)\n"
          << "  --vp-address=ADDR     VP register base (default: 0x1000)\n"
          << "  --hif-address=ADDR    HIF register base (default: 0x0800)\n"
          << "  --bridge-ip=ADDR      Bridge IP for FPGA (default: 10.0.0.1)\n"
          << "  --emulator            Using emulator (skip FPGA reset)\n"
          << "  --no-verify           Skip ILA response verification\n"
          << "  --forward             Forward (echo) mode: accept echoed "
             "requests\n";
      exit(0);
    }
  }
  return args;
}

// ============================================================================
// BRAM helpers
// ============================================================================

std::uint32_t bram_w_sample_addr() {
  std::uint32_t w = 0;
  while ((1u << w) < RAM_DEPTH)
    ++w;
  return w;
}

std::uint32_t load_le_u32(const std::uint8_t *p) {
  return std::uint32_t(p[0]) | (std::uint32_t(p[1]) << 8) |
         (std::uint32_t(p[2]) << 16) | (std::uint32_t(p[3]) << 24);
}

/// Build one RPC request for the increment handler.
/// Layout: [RPCHeader (24 bytes, ptp_timestamp zeroed)][data bytes...]
/// The FPGA overwrites header bytes 16-23 (ptp_timestamp field) with the PTP
/// send timestamp at transmit time.
std::vector<std::uint8_t> build_rpc_message(uint32_t msg_index,
                                            uint32_t payload_size) {
  using cudaq::realtime::fnv1a_hash;
  using cudaq::realtime::RPCHeader;

  constexpr uint32_t FUNC_ID = fnv1a_hash("rpc_increment");

  std::vector<std::uint8_t> msg(sizeof(RPCHeader) + payload_size, 0);
  auto *hdr = reinterpret_cast<RPCHeader *>(msg.data());
  hdr->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  hdr->function_id = FUNC_ID;
  hdr->arg_len = payload_size;
  hdr->request_id = msg_index;
  hdr->ptp_timestamp = 0;

  uint8_t *payload = msg.data() + sizeof(RPCHeader);
  for (uint32_t i = 0; i < payload_size; i++)
    payload[i] = static_cast<uint8_t>((msg_index + i) & 0xFF);

  return msg;
}

/// Pad message to 64-byte aligned window and write to BRAM.
void write_bram(hololink::Hololink &hl,
                const std::vector<std::vector<std::uint8_t>> &windows,
                std::size_t bytes_per_window) {
  if (bytes_per_window % 64 != 0)
    throw std::runtime_error("bytes_per_window must be a multiple of 64");

  std::size_t cycles = bytes_per_window / 64;
  if (cycles == 0)
    throw std::runtime_error("bytes_per_window is too small");

  if (windows.size() * cycles > RAM_DEPTH) {
    std::ostringstream msg;
    msg << "Requested " << windows.size() << " windows with " << cycles
        << " cycles each exceeds RAM depth " << RAM_DEPTH;
    throw std::runtime_error(msg.str());
  }

  const std::uint32_t w_sample_addr = bram_w_sample_addr();

  constexpr std::size_t kBatchWrites = 180;
  hololink::Hololink::WriteData write_data;

  for (std::size_t w = 0; w < windows.size(); ++w) {
    const auto &window = windows[w];
    for (std::size_t s = 0; s < cycles; ++s) {
      for (std::size_t i = 0; i < RAM_NUM; ++i) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t value = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          value = load_le_u32(window.data() + byte_offset);

        auto ram_addr = static_cast<std::uint32_t>(i << (w_sample_addr + 2));
        auto sample_addr = static_cast<std::uint32_t>((s + (w * cycles)) * 0x4);
        std::uint32_t address = RAM_ADDR + ram_addr + sample_addr;

        write_data.queue_write_uint32(address, value);
        if (write_data.size() >= kBatchWrites) {
          if (!hl.write_uint32(write_data))
            throw std::runtime_error("Failed to write BRAM batch");
          write_data = hololink::Hololink::WriteData();
        }
      }
    }
  }

  if (write_data.size() > 0) {
    if (!hl.write_uint32(write_data))
      throw std::runtime_error("Failed to write BRAM batch");
  }
}

/// Read back BRAM and verify contents.
bool verify_bram(hololink::Hololink &hl,
                 const std::vector<std::vector<std::uint8_t>> &windows,
                 std::size_t bytes_per_window) {
  const std::size_t cycles = bytes_per_window / 64;
  const auto total_cycles = static_cast<std::uint32_t>(windows.size() * cycles);
  const std::uint32_t w_sample_addr = bram_w_sample_addr();

  bool all_ok = true;
  std::size_t mismatches = 0;

  for (std::uint32_t i = 0; i < RAM_NUM; ++i) {
    std::uint32_t bank_base = RAM_ADDR + (i << (w_sample_addr + 2));
    auto [ok, readback] = hl.read_uint32(bank_base, total_cycles,
                                         hololink::Timeout::default_timeout());
    if (!ok) {
      std::cerr << "BRAM readback: failed to read bank " << i << "\n";
      return false;
    }

    for (std::size_t w = 0; w < windows.size(); ++w) {
      const auto &window = windows[w];
      for (std::size_t s = 0; s < cycles; ++s) {
        std::size_t word_index = s * RAM_NUM + i;
        std::size_t byte_offset = word_index * sizeof(std::uint32_t);
        std::uint32_t expected = 0;
        if (byte_offset + sizeof(std::uint32_t) <= window.size())
          expected = load_le_u32(window.data() + byte_offset);

        std::size_t sample_idx = w * cycles + s;
        std::uint32_t actual = readback[sample_idx];

        if (actual != expected) {
          if (mismatches < 10) {
            std::cerr << "  BRAM mismatch: bank=" << i
                      << " sample=" << sample_idx << " expected=0x" << std::hex
                      << expected << " got=0x" << actual << std::dec << "\n";
          }
          all_ok = false;
          ++mismatches;
        }
      }
    }
  }

  if (mismatches > 10)
    std::cerr << "  ... and " << (mismatches - 10) << " more mismatches\n";

  return all_ok;
}

// ============================================================================
// ILA functions
// ============================================================================

void ila_reset(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_RESET))
    throw std::runtime_error("ILA reset write failed");
  std::this_thread::sleep_for(std::chrono::seconds(1));
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable-after-reset write failed");
}

void ila_enable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_ENABLE))
    throw std::runtime_error("ILA enable write failed");
}

void ila_disable(hololink::Hololink &hl) {
  if (!hl.write_uint32(ILA_BASE_ADDR + ILA_CTRL_OFFSET, ILA_CTRL_DISABLE))
    throw std::runtime_error("ILA disable write failed");
}

std::uint32_t ila_sample_count(hololink::Hololink &hl) {
  return hl.read_uint32(ILA_BASE_ADDR + ILA_SAMPLE_ADDR_OFFSET);
}

/// Read captured ILA samples from RAM banks. Returns
/// vector of samples; each sample is ILA_NUM_RAM uint32 words (LSW-first).
std::vector<std::vector<std::uint32_t>> ila_dump(hololink::Hololink &hl,
                                                 std::uint32_t num_samples) {
  constexpr std::uint32_t ctrl_switch = 1u << (ILA_W_ADDR + 2 + ILA_W_RAM);
  // Max entries per block read: (1472 - 6 byte header) / 8 bytes per entry.
  // Use 128 for comfortable margin on both request and reply packets.
  constexpr std::uint32_t kChunkSize = 128;
  auto timeout = hololink::Timeout::default_timeout();

  std::vector<std::vector<std::uint32_t>> bank_data(ILA_NUM_RAM);
  for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y) {
    std::uint32_t bank_base =
        ILA_BASE_ADDR + ctrl_switch + (y << (ILA_W_ADDR + 2));
    bank_data[y].reserve(num_samples);
    for (std::uint32_t off = 0; off < num_samples; off += kChunkSize) {
      std::uint32_t n = std::min(kChunkSize, num_samples - off);
      auto [ok, data] = hl.read_uint32(bank_base + off * 4, n, timeout);
      if (!ok)
        throw std::runtime_error("Failed to read ILA bank " +
                                 std::to_string(y));
      bank_data[y].insert(bank_data[y].end(), data.begin(), data.end());
    }
  }

  std::vector<std::vector<std::uint32_t>> samples(
      num_samples, std::vector<std::uint32_t>(ILA_NUM_RAM));
  for (std::uint32_t i = 0; i < num_samples; ++i)
    for (std::uint32_t y = 0; y < ILA_NUM_RAM; ++y)
      samples[i][y] = bank_data[y][i];

  return samples;
}

/// Extract a single bit from a 585-bit ILA sample stored as 19 uint32 words.
bool get_bit(const std::vector<std::uint32_t> &sample, uint32_t bit_pos) {
  uint32_t word = bit_pos / 32;
  uint32_t offset = bit_pos % 32;
  return (sample[word] >> offset) & 1;
}

/// Extract the first 64 bytes (512 bits) of payload from an ILA sample.
std::vector<std::uint8_t>
extract_payload_bytes(const std::vector<std::uint32_t> &sample,
                      std::size_t num_bytes) {
  std::vector<std::uint8_t> bytes(num_bytes, 0);
  for (std::size_t i = 0; i < num_bytes && i < 64; ++i) {
    uint32_t word = sample[i / 4];
    bytes[i] = static_cast<std::uint8_t>((word >> ((i % 4) * 8)) & 0xFF);
  }
  return bytes;
}

/// Extract the 64-bit current_ptp_timestamp from ILA bits [584:521].
/// Returns {sec[31:0], nsec[31:0]} packed as a uint64.
std::uint64_t
extract_ila_ptp_timestamp(const std::vector<std::uint32_t> &sample) {
  // Bits 521..584 span words 16 and 17 (and partially 18).
  // bit 521 = word 16, offset 9
  // We need 64 bits starting at bit 521.
  uint64_t raw = 0;
  for (int b = 0; b < 64; ++b) {
    uint32_t bit_pos = 521 + b;
    uint32_t w = bit_pos / 32;
    uint32_t off = bit_pos % 32;
    if ((sample[w] >> off) & 1)
      raw |= (uint64_t(1) << b);
  }
  return raw;
}

/// Extract the echoed PTP send timestamp from RPCResponse.ptp_timestamp.
/// The dispatch kernel echoes this field from RPCHeader.ptp_timestamp.
std::uint64_t
extract_echoed_ptp_timestamp(const cudaq::realtime::RPCResponse *resp) {
  return resp->ptp_timestamp;
}

struct PtpTimestamp {
  uint32_t sec;
  uint32_t nsec;
};

PtpTimestamp decode_ptp(uint64_t raw) {
  // {sec[31:0], nsec[31:0]} -- sec in upper 32 bits, nsec in lower 32
  return {static_cast<uint32_t>(raw >> 32),
          static_cast<uint32_t>(raw & 0xFFFF'FFFF)};
}

/// Compute signed nanosecond difference (recv - send).
int64_t ptp_delta_ns(PtpTimestamp send, PtpTimestamp recv) {
  int64_t d_sec = static_cast<int64_t>(recv.sec) - send.sec;
  int64_t d_nsec = static_cast<int64_t>(recv.nsec) - send.nsec;
  return d_sec * 1'000'000'000LL + d_nsec;
}

} // namespace

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[]) {
  auto args = parse_args(argc, argv);

  std::cout << "=== Hololink Generic RPC Playback ===" << std::endl;
  std::cout << "Hololink: " << args.hololink_ip << std::endl;
  std::cout << "Messages: " << args.num_messages << std::endl;
  std::cout << "Payload size: " << args.payload_size << " bytes" << std::endl;

  // ------------------------------------------------------------------
  // Build Hololink DataChannel
  // ------------------------------------------------------------------
  hololink::Metadata channel_metadata;

  if (args.emulator) {
    channel_metadata["channel_ip"] = args.hololink_ip;
    channel_metadata["cpnx_ip"] = args.hololink_ip;
    channel_metadata["control_port"] =
        static_cast<std::int64_t>(args.control_port);
    channel_metadata["hsb_ip_version"] = static_cast<std::int64_t>(0x2501);
    channel_metadata["fpga_uuid"] = std::string("emulator");
    channel_metadata["serial_number"] = std::string("emulator-0");
    channel_metadata["peer_ip"] = args.hololink_ip;
    channel_metadata["vp_mask"] = static_cast<std::int64_t>(0x1);
    channel_metadata["data_plane"] = static_cast<std::int64_t>(0);
    channel_metadata["sensor"] = static_cast<std::int64_t>(0);
    channel_metadata["sif_address"] = static_cast<std::int64_t>(0);
    channel_metadata["vp_address"] = static_cast<std::int64_t>(args.vp_address);
    channel_metadata["hif_address"] =
        static_cast<std::int64_t>(args.hif_address);
  } else {
    channel_metadata = hololink::Enumerator::find_channel(args.hololink_ip);
    hololink::DataChannel::use_sensor(channel_metadata, 0);
  }

  hololink::DataChannel hololink_channel(channel_metadata);
  auto hololink = hololink_channel.hololink();

  hololink->start();
  if (!args.emulator) {
    hololink->reset();
  }

  // ------------------------------------------------------------------
  // Configure FPGA SIF for RDMA target
  // ------------------------------------------------------------------
  size_t frame_size = sizeof(cudaq::realtime::RPCHeader) + args.payload_size;
  size_t bytes_per_window = ((frame_size + 63) / 64) * 64;

  std::cout << "\n[1/4] Configuring RDMA target..." << std::endl;
  std::cout << "  Bridge QP: 0x" << std::hex << args.bridge_qp << std::dec
            << std::endl;
  std::cout << "  Bridge RKey: " << args.bridge_rkey << std::endl;
  std::cout << "  Bridge Buffer: 0x" << std::hex << args.bridge_buffer
            << std::dec << std::endl;
  std::cout << "  Page size: " << args.page_size << " bytes" << std::endl;
  std::cout << "  Num pages: " << args.num_pages << std::endl;
  std::cout << "  Frame size: " << bytes_per_window << " bytes" << std::endl;

  hololink_channel.authenticate(args.bridge_qp, args.bridge_rkey);
  hololink_channel.configure_roce(
      args.bridge_buffer, static_cast<uint32_t>(bytes_per_window),
      static_cast<uint32_t>(args.page_size), args.num_pages, ROCEV2_UDP_PORT);

  std::cout << "  RDMA target configured" << std::endl;

  // ------------------------------------------------------------------
  // Disable player, configure registers, load BRAM
  // ------------------------------------------------------------------
  std::cout << "\n[2/4] Loading RPC messages into BRAM..." << std::endl;

  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_DISABLE))
    throw std::runtime_error("Failed to disable player");

  hololink::Hololink::WriteData config_write;
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_WINDOW_SIZE_OFFSET,
                                  static_cast<std::uint32_t>(bytes_per_window));
  config_write.queue_write_uint32(
      PLAYER_ADDR + PLAYER_WINDOW_NUMBER_OFFSET,
      static_cast<std::uint32_t>(args.num_messages));
  config_write.queue_write_uint32(PLAYER_ADDR + PLAYER_TIMER_OFFSET,
                                  RF_SOC_TIMER_SCALE *
                                      DEFAULT_TIMER_SPACING_US);
  if (!hololink->write_uint32(config_write))
    throw std::runtime_error("Failed to configure player");

  // Build and load RPC messages
  std::vector<std::vector<std::uint8_t>> windows;
  windows.reserve(args.num_messages);
  for (uint32_t i = 0; i < args.num_messages; i++) {
    auto msg = build_rpc_message(i, args.payload_size);
    msg.resize(bytes_per_window, 0); // pad to window boundary
    windows.push_back(std::move(msg));
  }

  write_bram(*hololink, windows, bytes_per_window);
  std::cout << "  BRAM write completed (" << args.num_messages << " messages)"
            << std::endl;

  // Verify BRAM contents
  std::cout << "  Verifying BRAM..." << std::endl;
  if (!verify_bram(*hololink, windows, bytes_per_window)) {
    std::cerr << "  BRAM readback verification FAILED" << std::endl;
  } else {
    std::cout << "  BRAM readback verification PASSED" << std::endl;
  }

  // ------------------------------------------------------------------
  // Arm ILA and trigger playback
  // ------------------------------------------------------------------
  std::cout << "\n[3/4] Triggering playback..." << std::endl;

  if (args.verify) {
    ila_disable(*hololink);
    ila_reset(*hololink);
    ila_enable(*hololink);
    std::cout << "  ILA: armed for capture" << std::endl;
  }

  // Disable metadata packet (set bit 16 of METADATA_PACKET_ADDR via RMW)
  // Needed for FPGA bitfile 0x0227+; comment out for older bitfiles (e.g.
  // 0x2601).
  {
    std::uint32_t val = hololink->read_uint32(METADATA_PACKET_ADDR);
    if (!hololink->write_uint32(METADATA_PACKET_ADDR, val | (1u << 16)))
      throw std::runtime_error("Failed to disable metadata packet");
  }

  // Set SIF TX streaming threshold to zero for immediate streaming.
  if (!hololink->write_uint32(SIF_TX_THRESHOLD_ADDR,
                              SIF_TX_THRESHOLD_IMMEDIATE))
    throw std::runtime_error("Failed to set SIF TX streaming threshold");

  // Enable player in single-pass mode
  if (!hololink->write_uint32(PLAYER_ADDR + PLAYER_ENABLE_OFFSET,
                              PLAYER_ENABLE_SINGLEPASS))
    throw std::runtime_error("Failed to enable player");

  std::cout << "  Playback triggered: " << args.num_messages << " messages"
            << std::endl;

  // ------------------------------------------------------------------
  // Wait and verify ILA capture
  // ------------------------------------------------------------------
  if (args.verify) {
    std::cout << "\n[4/4] Verifying responses..." << std::endl;

    constexpr int kStableChecks = 2;
    constexpr int kPollIntervalMs = 500;
    constexpr int kVerifyTimeoutMs = 30000;
    std::cout << "  Waiting for ILA capture to stabilize (timeout "
              << kVerifyTimeoutMs << " ms)..." << std::endl;

    std::uint32_t prev_count = 0;
    int stable = 0;
    int elapsed = 0;
    while (elapsed < kVerifyTimeoutMs) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
      elapsed += kPollIntervalMs;
      std::uint32_t count = ila_sample_count(*hololink);
      if (count > 0 && count == prev_count)
        ++stable;
      else
        stable = 0;
      prev_count = count;
      if (stable >= kStableChecks)
        break;
    }

    std::uint32_t actual_samples = ila_sample_count(*hololink);
    ila_disable(*hololink);

    if (actual_samples == 0) {
      std::cerr << "  ILA: captured 0 samples (timeout " << kVerifyTimeoutMs
                << " ms)" << std::endl;
      return 1;
    }
    std::cout << "  ILA: captured " << actual_samples << " samples"
              << std::endl;

    // Read all captured samples
    auto samples = ila_dump(*hololink, actual_samples);
    std::cout << "  Read " << samples.size() << " samples from ILA"
              << std::endl;

    uint32_t matched = 0;
    uint32_t header_errors = 0;
    uint32_t payload_errors = 0;
    uint32_t tvalid_zero = 0;
    uint32_t rpc_responses = 0;
    uint32_t non_rpc_frames = 0;
    std::set<uint32_t> seen_request_ids;

    int64_t lat_min = std::numeric_limits<int64_t>::max();
    int64_t lat_max = std::numeric_limits<int64_t>::min();
    int64_t lat_sum = 0;
    uint32_t lat_count = 0;

    struct LatencySample {
      uint32_t shot;
      uint32_t send_sec, send_nsec;
      uint32_t recv_sec, recv_nsec;
      int64_t delta_ns;
    };
    std::vector<LatencySample> lat_samples;

    for (auto &sample : samples) {
      if (!get_bit(sample, ILA_TVALID_BIT)) {
        ++tvalid_zero;
        continue;
      }

      auto payload = extract_payload_bytes(sample, 64);

      auto *resp = reinterpret_cast<const cudaq::realtime::RPCResponse *>(
          payload.data());

      uint32_t expected_magic = args.forward
                                    ? cudaq::realtime::RPC_MAGIC_REQUEST
                                    : cudaq::realtime::RPC_MAGIC_RESPONSE;

      if (resp->magic != expected_magic) {
        ++non_rpc_frames;
        continue;
      }

      ++rpc_responses;

      if (!args.forward && resp->status != 0) {
        std::cerr << "  Response request_id=" << resp->request_id
                  << ": error status " << resp->status << std::endl;
        ++header_errors;
        continue;
      }

      uint32_t rid = resp->request_id;
      seen_request_ids.insert(rid);

      if (!args.forward) {
        const uint8_t *result_data =
            payload.data() + sizeof(cudaq::realtime::RPCResponse);
        bool ok = true;
        uint32_t check_len = std::min(resp->result_len, args.payload_size);

        for (uint32_t j = 0; j < check_len && ok; j++) {
          uint8_t expected = static_cast<uint8_t>(((rid + j) & 0xFF) + 1);
          if (result_data[j] != expected) {
            if (payload_errors < 5) {
              std::cerr << "  Shot " << rid << " byte " << j << ": expected "
                        << (int)expected << " got " << (int)result_data[j]
                        << std::endl;
            }
            ok = false;
          }
        }

        if (ok)
          ++matched;
        else
          ++payload_errors;
      } else {
        ++matched;
      }

      // PTP round-trip latency: send timestamp from response header,
      // receive timestamp from ILA bits [584:521].
      {
        uint64_t send_raw = extract_echoed_ptp_timestamp(resp);
        uint64_t recv_raw = extract_ila_ptp_timestamp(sample);
        if (send_raw != 0 && recv_raw != 0) {
          auto send_ts = decode_ptp(send_raw);
          auto recv_ts = decode_ptp(recv_raw);
          int64_t delta = ptp_delta_ns(send_ts, recv_ts);
          lat_sum += delta;
          ++lat_count;
          if (delta < lat_min)
            lat_min = delta;
          if (delta > lat_max)
            lat_max = delta;

          lat_samples.push_back({rid, send_ts.sec, send_ts.nsec, recv_ts.sec,
                                 recv_ts.nsec, delta});

          if (lat_count <= 5) {
            std::cout << "  Msg " << std::setw(3) << rid
                      << ": send={sec=" << send_ts.sec
                      << ", nsec=" << send_ts.nsec
                      << "} recv={sec=" << recv_ts.sec
                      << ", nsec=" << recv_ts.nsec << "} delta=" << delta
                      << " ns" << std::endl;
          }
        }
      }
    }

    std::cout << "\n=== Verification Summary ===" << std::endl;
    std::cout << "  ILA samples captured:   " << actual_samples << std::endl;
    std::cout << "  tvalid=0 (idle):        " << tvalid_zero << std::endl;
    std::cout << "  RPC responses:          " << rpc_responses << std::endl;
    std::cout << "  Non-RPC frames:         " << non_rpc_frames << std::endl;
    std::cout << "  Unique messages verified: " << seen_request_ids.size()
              << " of " << args.num_messages << std::endl;
    std::cout << "  Responses matched:    " << matched << std::endl;
    std::cout << "  Header errors:        " << header_errors << std::endl;
    std::cout << "  Payload errors:       " << payload_errors << std::endl;

    if (lat_count > 0) {
      double lat_avg = static_cast<double>(lat_sum) / lat_count;
      std::cout << "\n=== PTP Round-Trip Latency ===" << std::endl;
      std::cout << "  Samples:  " << lat_count << std::endl;
      std::cout << "  Min:      " << lat_min << " ns" << std::endl;
      std::cout << "  Max:      " << lat_max << " ns" << std::endl;
      std::cout << "  Avg:      " << std::fixed << std::setprecision(1)
                << lat_avg << " ns" << std::endl;
      const std::string csv_path = "ptp_latency.csv";
      std::ofstream csv(csv_path);
      if (csv.is_open()) {
        csv << "shot,send_sec,send_nsec,recv_sec,recv_nsec,delta_ns\n";
        for (auto &s : lat_samples)
          csv << s.shot << "," << s.send_sec << "," << s.send_nsec << ","
              << s.recv_sec << "," << s.recv_nsec << "," << s.delta_ns << "\n";
        csv.close();
        std::cout << "  CSV written: " << csv_path << std::endl;
      }
    } else {
      std::cout << "\n  PTP latency: no valid timestamps found" << std::endl;
    }

    if (payload_errors == 0 && header_errors == 0 &&
        seen_request_ids.size() > 0) {
      std::cout << "  RESULT: PASS" << std::endl;
      return 0;
    } else {
      std::cout << "  RESULT: FAIL" << std::endl;
      return 1;
    }
  } else {
    std::cout << "\n[4/4] Verification skipped (--no-verify)" << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << "\n*** PLAYBACK COMPLETE ***" << std::endl;
    return 0;
  }
}
