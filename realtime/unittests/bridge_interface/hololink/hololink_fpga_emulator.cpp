/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_fpga_emulator.cpp
/// @brief Software FPGA emulator for Hololink RPC testing.
///
/// Emulates the FPGA's role in the RPC pipeline:
///   1. Hololink UDP control plane server (register read/write)
///   2. Playback BRAM (receives payloads from playback tool)
///   3. RDMA transmit (sends RPC requests to bridge)
///   4. RDMA receive (receives RPC responses from bridge)
///   5. ILA capture RAM (stores responses for verification readback)
///
/// Three-tool workflow:
///   1. Start this emulator (prints QP number)
///   2. Start hololink_mock_decoder_bridge with --remote-qp=<emulator_qp>
///   3. Start hololink_fpga_syndrome_playback --control-port=<port>
///      with bridge's QP/RKEY/buffer-addr
///
/// The playback tool drives the emulator via UDP just as it would a real FPGA.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

//==============================================================================
// Global shutdown flag
//==============================================================================

static std::atomic<bool> g_shutdown{false};
static void signal_handler(int) { g_shutdown = true; }

//==============================================================================
// Hololink Protocol Constants
//==============================================================================

static constexpr uint8_t WR_DWORD = 0x04;
static constexpr uint8_t WR_BLOCK = 0x09;
static constexpr uint8_t RD_DWORD = 0x14;
static constexpr uint8_t RD_BLOCK = 0x19;

static constexpr uint8_t REQUEST_FLAGS_ACK_REQUEST = 0x01;
static constexpr uint8_t RESPONSE_SUCCESS = 0x00;

// VP register offsets (relative to vp_address)
static constexpr uint32_t DP_QP = 0x00;
static constexpr uint32_t DP_RKEY = 0x04;
static constexpr uint32_t DP_PAGE_LSB = 0x08;
static constexpr uint32_t DP_PAGE_MSB = 0x0C;
static constexpr uint32_t DP_PAGE_INC = 0x10;
static constexpr uint32_t DP_MAX_BUFF = 0x14;
static constexpr uint32_t DP_BUFFER_LENGTH = 0x18;

// HIF register offsets (relative to hif_address)
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
static constexpr int BRAM_W_SAMPLE_ADDR = 9; // log2(512 entries)
static constexpr int BRAM_BANK_STRIDE = 1 << (BRAM_W_SAMPLE_ADDR + 2); // 2048

// ILA capture
static constexpr uint32_t ILA_BASE = 0x40000000;
static constexpr uint32_t ILA_CTRL = ILA_BASE + 0x00;
static constexpr uint32_t ILA_STATUS = ILA_BASE + 0x80;
static constexpr uint32_t ILA_SAMPLE_ADDR = ILA_BASE + 0x84;
static constexpr uint32_t ILA_DATA_BASE = 0x40100000;
static constexpr int ILA_NUM_BANKS = 17;
static constexpr int ILA_W_ADDR = 13; // log2(8192 entries)
static constexpr int ILA_BANK_STRIDE = 1 << (ILA_W_ADDR + 2); // 32768

// Ring buffer
static constexpr int NUM_BUFFERS = 64;

//==============================================================================
// RDMA Context (adapted from cuda-qx rdma_utils.hpp)
//==============================================================================

class RdmaContext {
public:
  ~RdmaContext() { cleanup(); }

  bool open(const std::string &device_name, int port = 1) {
    int num_devices;
    ibv_device **devices = ibv_get_device_list(&num_devices);
    if (!devices || num_devices == 0)
      return false;

    ibv_device *target = nullptr;
    for (int i = 0; i < num_devices; i++) {
      if (device_name == ibv_get_device_name(devices[i])) {
        target = devices[i];
        break;
      }
    }
    if (!target) {
      ibv_free_device_list(devices);
      return false;
    }

    ctx_ = ibv_open_device(target);
    ibv_free_device_list(devices);
    if (!ctx_)
      return false;

    port_ = port;
    pd_ = ibv_alloc_pd(ctx_);
    if (!pd_) {
      cleanup();
      return false;
    }

    if (ibv_query_port(ctx_, port_, &port_attr_) != 0) {
      cleanup();
      return false;
    }

    gid_index_ = find_roce_v2_gid_index();
    return true;
  }

  ibv_cq *create_cq(int size) {
    return ibv_create_cq(ctx_, size, nullptr, nullptr, 0);
  }

  ibv_mr *register_memory(void *addr, size_t size,
                          int access = IBV_ACCESS_LOCAL_WRITE |
                                       IBV_ACCESS_REMOTE_WRITE) {
    return ibv_reg_mr(pd_, addr, size, access);
  }

  ibv_qp *create_qp(ibv_cq *send_cq, ibv_cq *recv_cq, uint32_t max_send_wr = 64,
                    uint32_t max_recv_wr = 64) {
    ibv_qp_init_attr init_attr{};
    init_attr.qp_type = IBV_QPT_UC; // Unreliable Connected - matches FPGA
    init_attr.send_cq = send_cq;
    init_attr.recv_cq = recv_cq;
    init_attr.cap.max_send_wr = max_send_wr;
    init_attr.cap.max_recv_wr = max_recv_wr;
    init_attr.cap.max_send_sge = 1;
    init_attr.cap.max_recv_sge = 1;
    return ibv_create_qp(pd_, &init_attr);
  }

  bool qp_to_init(ibv_qp *qp) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = port_;
    attr.pkey_index = 0;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX |
                             IBV_QP_ACCESS_FLAGS) == 0;
  }

  bool qp_to_rtr(ibv_qp *qp, const ibv_gid &remote_gid, uint32_t remote_qp_num,
                 uint32_t psn = 0) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = port_attr_.active_mtu;
    attr.dest_qp_num = remote_qp_num;
    attr.rq_psn = psn;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = remote_gid;
    attr.ah_attr.grh.sgid_index = gid_index_;
    attr.ah_attr.grh.hop_limit = 64;
    attr.ah_attr.grh.traffic_class = 0;
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port_;
    return ibv_modify_qp(qp, &attr,
                         IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                             IBV_QP_RQ_PSN | IBV_QP_AV) == 0;
  }

  bool qp_to_rts(ibv_qp *qp, uint32_t psn = 0) {
    ibv_qp_attr attr{};
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = psn;
    return ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN) == 0;
  }

  bool post_recv(ibv_qp *qp, uint64_t wr_id, void *addr, uint32_t length,
                 uint32_t lkey) {
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(addr);
    sge.length = length;
    sge.lkey = lkey;

    ibv_recv_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.next = nullptr;

    ibv_recv_wr *bad_wr = nullptr;
    return ibv_post_recv(qp, &wr, &bad_wr) == 0;
  }

  bool post_rdma_write_imm(ibv_qp *qp, uint64_t wr_id, void *local_addr,
                           uint32_t length, uint32_t lkey, uint64_t remote_addr,
                           uint32_t rkey, uint32_t imm_data) {
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(local_addr);
    sge.length = length;
    sge.lkey = lkey;

    ibv_send_wr wr{};
    wr.wr_id = wr_id;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.imm_data = htonl(imm_data);
    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey = rkey;
    wr.next = nullptr;

    ibv_send_wr *bad_wr = nullptr;
    return ibv_post_send(qp, &wr, &bad_wr) == 0;
  }

  int poll_cq(ibv_cq *cq, ibv_wc *wc, int max_wc = 1) {
    return ibv_poll_cq(cq, max_wc, wc);
  }

  int get_gid_index() const { return gid_index_; }

private:
  void cleanup() {
    if (pd_) {
      ibv_dealloc_pd(pd_);
      pd_ = nullptr;
    }
    if (ctx_) {
      ibv_close_device(ctx_);
      ctx_ = nullptr;
    }
  }

  int find_roce_v2_gid_index() {
    int best_gid = -1;
    for (int i = 0; i < port_attr_.gid_tbl_len; i++) {
      ibv_gid gid;
      if (ibv_query_gid(ctx_, port_, i, &gid) == 0) {
        if (gid.raw[10] == 0xff && gid.raw[11] == 0xff) {
          best_gid = i; // Last match = RoCE v2
        }
      }
    }
    return (best_gid >= 0) ? best_gid : 0;
  }

  ibv_context *ctx_ = nullptr;
  ibv_pd *pd_ = nullptr;
  ibv_port_attr port_attr_{};
  int port_ = 1;
  int gid_index_ = 0;
};

//==============================================================================
// RDMA Buffer
//==============================================================================

class RdmaBuffer {
public:
  ~RdmaBuffer() { release(); }

  bool allocate(RdmaContext &ctx, size_t size) {
    size_t page_size = 4096;
    size_t aligned = ((size + page_size - 1) / page_size) * page_size;
    data_ = aligned_alloc(page_size, aligned);
    if (!data_)
      return false;
    size_ = size;
    memset(data_, 0, aligned);
    mr_ = ctx.register_memory(data_, aligned,
                              IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr_) {
      ::free(data_);
      data_ = nullptr;
      return false;
    }
    return true;
  }

  void release() {
    if (mr_) {
      ibv_dereg_mr(mr_);
      mr_ = nullptr;
    }
    if (data_) {
      ::free(data_);
      data_ = nullptr;
    }
  }

  void *data() const { return data_; }
  size_t size() const { return size_; }
  uint32_t lkey() const { return mr_ ? mr_->lkey : 0; }
  uint32_t rkey() const { return mr_ ? mr_->rkey : 0; }

private:
  void *data_ = nullptr;
  size_t size_ = 0;
  ibv_mr *mr_ = nullptr;
};

//==============================================================================
// Emulated Register File
//==============================================================================

class RegisterFile {
public:
  void write(uint32_t addr, uint32_t value) {
    std::lock_guard<std::mutex> lock(mu_);
    regs_[addr] = value;
  }

  uint32_t read(uint32_t addr) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = regs_.find(addr);
    return (it != regs_.end()) ? it->second : 0;
  }

  /// Batch write (for BRAM loading efficiency).
  void write_batch(const std::vector<std::pair<uint32_t, uint32_t>> &writes) {
    std::lock_guard<std::mutex> lock(mu_);
    for (auto &[addr, val] : writes) {
      regs_[addr] = val;
    }
  }

  /// Read a range of contiguous 32-bit registers.
  std::vector<uint32_t> read_range(uint32_t base_addr, uint32_t count) const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<uint32_t> result(count);
    for (uint32_t i = 0; i < count; i++) {
      auto it = regs_.find(base_addr + i * 4);
      result[i] = (it != regs_.end()) ? it->second : 0;
    }
    return result;
  }

private:
  mutable std::mutex mu_;
  std::unordered_map<uint32_t, uint32_t> regs_;
};

//==============================================================================
// RDMA Target Config (decoded from VP register writes)
//==============================================================================

struct RdmaTargetConfig {
  uint32_t qp_number = 0;
  uint32_t rkey = 0;
  uint64_t buffer_addr = 0;
  uint32_t page_inc = 0; // bytes
  uint32_t max_buff = 0; // max buffer index
  uint32_t buffer_length = 0;

  // Temporary storage for two-part address
  uint32_t page_lsb = 0;
  uint32_t page_msb = 0;

  // Track whether key fields were explicitly set (buffer_addr=0 is valid
  // when Hololink uses IOVA with dmabuf).
  bool qp_set = false;
  bool rkey_set = false;

  void update_addr() {
    // Hololink encodes: PAGE_LSB = addr >> 7, PAGE_MSB = addr >> 32
    // Reconstruct: addr = (MSB << 32) | (LSB << 7)
    buffer_addr = (static_cast<uint64_t>(page_msb) << 32) |
                  (static_cast<uint64_t>(page_lsb) << 7);
  }

  bool is_complete() const {
    // buffer_addr=0 is valid (Hololink IOVA/dmabuf), so we only check
    // that QP and RKEY were explicitly set.
    return qp_set && rkey_set;
  }

  void print() const {
    std::cout << "  RDMA Target Config:" << std::endl;
    std::cout << "    QP: 0x" << std::hex << qp_number << std::dec << std::endl;
    std::cout << "    RKEY: 0x" << std::hex << rkey << std::dec << std::endl;
    std::cout << "    Buffer addr: 0x" << std::hex << buffer_addr << std::dec
              << std::endl;
    std::cout << "    Page inc: " << page_inc << " bytes" << std::endl;
    std::cout << "    Max buff: " << max_buff << std::endl;
  }
};

//==============================================================================
// UDP Control Plane Server
//==============================================================================

class ControlPlaneServer {
public:
  ControlPlaneServer(uint16_t port, uint32_t vp_address, uint32_t hif_address,
                     RegisterFile &regs)
      : port_(port), vp_addr_(vp_address), hif_addr_(hif_address), regs_(regs) {
  }

  ~ControlPlaneServer() { stop(); }

  void set_my_qp(uint32_t qp) { my_qp_ = qp; }

  bool start() {
    fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0)
      return false;

    int opt = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port_);
    if (bind(fd_, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
      ::close(fd_);
      fd_ = -1;
      return false;
    }

    running_ = true;
    thread_ = std::thread(&ControlPlaneServer::run, this);
    return true;
  }

  void stop() {
    running_ = false;
    if (fd_ >= 0) {
      shutdown(fd_, SHUT_RDWR);
      ::close(fd_);
      fd_ = -1;
    }
    if (thread_.joinable())
      thread_.join();
  }

  /// Block until RDMA config is complete or timeout.
  bool wait_for_config(int timeout_ms = 60000) {
    auto start = std::chrono::steady_clock::now();
    while (!target_.is_complete() && !g_shutdown) {
      auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count();
      if (elapsed >= timeout_ms)
        return false;
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return target_.is_complete();
  }

  const RdmaTargetConfig &target() const { return target_; }

  /// Check if player_enable was set to 1.
  bool playback_triggered() const { return playback_triggered_.load(); }
  void clear_playback_trigger() { playback_triggered_ = false; }

  /// Get player config.
  uint32_t window_size() const { return regs_.read(PLAYER_WIN_SIZE); }
  uint32_t window_number() const { return regs_.read(PLAYER_WIN_NUM); }
  uint32_t timer_spacing() const { return regs_.read(PLAYER_TIMER); }

private:
  void run() {
    std::vector<uint8_t> buf(4096);
    while (running_ && !g_shutdown) {
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(fd_, &fds);
      timeval tv{0, 100000}; // 100ms

      int ready = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
      if (ready <= 0)
        continue;

      sockaddr_in client{};
      socklen_t clen = sizeof(client);
      ssize_t len = recvfrom(fd_, buf.data(), buf.size(), 0,
                             reinterpret_cast<sockaddr *>(&client), &clen);
      if (len < 6)
        continue;

      handle_packet(buf.data(), static_cast<size_t>(len), client);
    }
  }

  // --- Packet helpers ---

  static uint32_t read_be32(const uint8_t *p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
           (uint32_t(p[2]) << 8) | p[3];
  }

  static uint16_t read_be16(const uint8_t *p) {
    return (uint16_t(p[0]) << 8) | p[1];
  }

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

  // --- Handle incoming packet ---

  void handle_packet(const uint8_t *data, size_t len,
                     const sockaddr_in &client) {
    uint8_t opcode = data[0];
    uint8_t flags = data[1];
    uint16_t seq = read_be16(data + 2);

    switch (opcode) {
    case WR_DWORD:
      if (len >= 14)
        handle_wr_dword(data, flags, seq, client);
      break;
    case WR_BLOCK:
      handle_wr_block(data, len, flags, seq, client);
      break;
    case RD_DWORD:
      if (len >= 10)
        handle_rd_dword(data, flags, seq, client);
      break;
    case RD_BLOCK:
      handle_rd_block(data, len, flags, seq, client);
      break;
    default:
      // Unknown opcode - send error ACK
      if (flags & REQUEST_FLAGS_ACK_REQUEST)
        send_write_ack(client, opcode, flags, seq);
      break;
    }
  }

  void handle_wr_dword(const uint8_t *data, uint8_t flags, uint16_t seq,
                       const sockaddr_in &client) {
    uint32_t addr = read_be32(data + 6);
    uint32_t val = read_be32(data + 10);
    process_register_write(addr, val);
    if (flags & REQUEST_FLAGS_ACK_REQUEST)
      send_write_ack(client, WR_DWORD, flags, seq);
  }

  void handle_wr_block(const uint8_t *data, size_t len, uint8_t flags,
                       uint16_t seq, const sockaddr_in &client) {
    // Pairs start at offset 6, each pair is 8 bytes
    size_t offset = 6;
    std::vector<std::pair<uint32_t, uint32_t>> batch;
    while (offset + 8 <= len) {
      uint32_t addr = read_be32(data + offset);
      uint32_t val = read_be32(data + offset + 4);
      batch.push_back({addr, val});
      offset += 8;
    }

    // Batch write to register file
    regs_.write_batch(batch);

    // Process VP register updates
    for (auto &[addr, val] : batch) {
      process_vp_update(addr, val);
      check_player_enable(addr, val);
    }

    if (flags & REQUEST_FLAGS_ACK_REQUEST)
      send_write_ack(client, WR_BLOCK, flags, seq);
  }

  void handle_rd_dword(const uint8_t *data, uint8_t flags, uint16_t seq,
                       const sockaddr_in &client) {
    uint32_t addr = read_be32(data + 6);
    uint32_t val = regs_.read(addr);

    // Response: cmd(1) + flags(1) + seq(2) + response_code(1) + reserved(1) +
    // addr(4) + value(4) + latched_seq(2) = 16 bytes
    uint8_t resp[16];
    resp[0] = RD_DWORD;
    resp[1] = flags;
    write_be16(resp + 2, seq);
    resp[4] = RESPONSE_SUCCESS;
    resp[5] = 0; // reserved
    write_be32(resp + 6, addr);
    write_be32(resp + 10, val);
    write_be16(resp + 14, seq); // latched sequence

    sendto(fd_, resp, sizeof(resp), 0,
           reinterpret_cast<const sockaddr *>(&client), sizeof(client));
  }

  void handle_rd_block(const uint8_t *data, size_t len, uint8_t flags,
                       uint16_t seq, const sockaddr_in &client) {
    // Parse addresses from request
    std::vector<uint32_t> addrs;
    size_t offset = 6;
    while (offset + 8 <= len) {
      addrs.push_back(read_be32(data + offset));
      offset += 8;
    }

    // Build response: cmd(1) + flags(1) + seq(2) + rc(1) + reserved(1) +
    //                 N*(addr(4)+value(4)) + latched_seq(2)
    size_t resp_len = 6 + addrs.size() * 8 + 2;
    std::vector<uint8_t> resp(resp_len);
    resp[0] = RD_BLOCK;
    resp[1] = flags;
    write_be16(resp.data() + 2, seq);
    resp[4] = RESPONSE_SUCCESS;
    resp[5] = 0;

    size_t roff = 6;
    for (uint32_t a : addrs) {
      uint32_t val = regs_.read(a);
      write_be32(resp.data() + roff, a);
      write_be32(resp.data() + roff + 4, val);
      roff += 8;
    }
    write_be16(resp.data() + roff, seq); // latched sequence

    sendto(fd_, resp.data(), resp.size(), 0,
           reinterpret_cast<const sockaddr *>(&client), sizeof(client));
  }

  // --- Write ACK for WR_DWORD / WR_BLOCK ---

  void send_write_ack(const sockaddr_in &client, uint8_t cmd, uint8_t flags,
                      uint16_t seq) {
    uint8_t resp[5];
    resp[0] = cmd;
    resp[1] = flags;
    write_be16(resp + 2, seq);
    resp[4] = RESPONSE_SUCCESS;
    sendto(fd_, resp, sizeof(resp), 0,
           reinterpret_cast<const sockaddr *>(&client), sizeof(client));
  }

  // --- Register write processing ---

  void process_register_write(uint32_t addr, uint32_t val) {
    regs_.write(addr, val);
    process_vp_update(addr, val);
    check_player_enable(addr, val);
  }

  void process_vp_update(uint32_t addr, uint32_t val) {
    // Check if this is a VP register (relative to vp_addr_)
    if (addr < vp_addr_ || addr >= vp_addr_ + 0x100)
      return;

    uint32_t offset = addr - vp_addr_;
    switch (offset) {
    case DP_QP:
      target_.qp_number = val;
      target_.qp_set = true;
      break;
    case DP_RKEY:
      target_.rkey = val;
      target_.rkey_set = true;
      break;
    case DP_PAGE_LSB:
      target_.page_lsb = val;
      target_.update_addr();
      break;
    case DP_PAGE_MSB:
      target_.page_msb = val;
      target_.update_addr();
      break;
    case DP_PAGE_INC:
      target_.page_inc = val << 7; // PAGES encoding: value * 128
      break;
    case DP_MAX_BUFF:
      target_.max_buff = val;
      break;
    case DP_BUFFER_LENGTH:
      target_.buffer_length = val;
      break;
    }
  }

  void check_player_enable(uint32_t addr, uint32_t val) {
    if (addr == PLAYER_ENABLE && val == 1) {
      playback_triggered_ = true;
    }
  }

  uint16_t port_;
  uint32_t vp_addr_;
  uint32_t hif_addr_;
  RegisterFile &regs_;
  int fd_ = -1;
  std::atomic<bool> running_{false};
  std::thread thread_;
  uint32_t my_qp_ = 0;
  RdmaTargetConfig target_;
  std::atomic<bool> playback_triggered_{false};
};

//==============================================================================
// BRAM Reassembly
//==============================================================================

/// Reassemble one window from the 16-bank BRAM layout.
/// Each 64-byte beat is spread across 16 banks (4 bytes each).
/// @param regs Register file to read from
/// @param window_index Window number
/// @param cycles_per_window Number of 64-byte beats per window
/// @return Reassembled window payload
static std::vector<uint8_t> reassemble_window(const RegisterFile &regs,
                                              uint32_t window_index,
                                              uint32_t cycles_per_window) {
  std::vector<uint8_t> payload(cycles_per_window * 64, 0);
  for (uint32_t cycle = 0; cycle < cycles_per_window; cycle++) {
    uint32_t sample_index = window_index * cycles_per_window + cycle;
    for (int bank = 0; bank < BRAM_NUM_BANKS; bank++) {
      uint32_t addr =
          RAM_BASE + (bank << (BRAM_W_SAMPLE_ADDR + 2)) + (sample_index * 4);
      uint32_t val = regs.read(addr);
      // Store as little-endian (matching FPGA BRAM word order)
      size_t byte_offset = cycle * 64 + bank * 4;
      memcpy(&payload[byte_offset], &val, 4);
    }
  }
  return payload;
}

//==============================================================================
// ILA Capture Storage
//==============================================================================

/// Store a correction response into the ILA capture register file.
/// The ILA stores each sample across 17 banks of 32-bit words.
/// Banks 0-15 = 512-bit AXI data bus (raw correction bytes).
/// Bank 16    = control signals:
///   bit 0 = tvalid (bit 512 of the captured word)
///   bit 1 = tlast  (bit 513)
///   bits [8:2] = wr_tcnt (bits 520:514, 7-bit write transaction count)
static void store_ila_sample(RegisterFile &regs, uint32_t sample_index,
                             const uint8_t *data, size_t data_len) {
  // Spread the data across banks 0-15 (the 512-bit AXI data bus).
  for (int bank = 0; bank < ILA_NUM_BANKS - 1; bank++) {
    uint32_t addr =
        ILA_DATA_BASE + (bank << (ILA_W_ADDR + 2)) + (sample_index * 4);
    uint32_t val = 0;
    size_t byte_offset = bank * 4;
    if (byte_offset < data_len) {
      size_t copy_len = std::min<size_t>(4, data_len - byte_offset);
      memcpy(&val, data + byte_offset, copy_len);
    }
    regs.write(addr, val);
  }

  // Bank 16: set control signals (tvalid=1, tlast=1, wr_tcnt=1)
  {
    uint32_t ctrl_addr = ILA_DATA_BASE +
                         ((ILA_NUM_BANKS - 1) << (ILA_W_ADDR + 2)) +
                         (sample_index * 4);
    uint32_t ctrl_val = 0;
    ctrl_val |= (1u << 0); // tvalid (bit 512)
    ctrl_val |= (1u << 1); // tlast  (bit 513)
    ctrl_val |= (1u << 2); // wr_tcnt = 1 (bits 514+, value 1 in 7-bit field)
    regs.write(ctrl_addr, ctrl_val);
  }

  // Update sample count
  regs.write(ILA_SAMPLE_ADDR, sample_index + 1);
}

//==============================================================================
// Command-Line Arguments
//==============================================================================

struct EmulatorArgs {
  std::string device = "rocep1s0f0";
  int ib_port = 1;
  uint16_t control_port = 8193;
  std::string bridge_ip = ""; // Bridge IP (for GID, auto-detect if empty)
  uint32_t vp_address = 0x1000;
  uint32_t hif_address = 0x0800;
  size_t page_size = 256; // Default slot size for responses RX
};

static void print_usage(const char *prog) {
  std::cout
      << "Usage: " << prog << " [options]\n"
      << "\nFPGA emulator for QEC decode loop testing.\n"
      << "\nOptions:\n"
      << "  --device=NAME         IB device name (default: rocep1s0f0)\n"
      << "  --ib-port=N           IB port number (default: 1)\n"
      << "  --port=N              UDP control plane port (default: 8193)\n"
      << "  --bridge-ip=ADDR      Bridge tool IP for GID (default: auto)\n"
      << "  --vp-address=ADDR     VP register base (default: 0x1000)\n"
      << "  --hif-address=ADDR    HIF register base (default: 0x0800)\n"
      << "  --page-size=N         Slot size for correction RX (default: 256)\n"
      << "  --help                Show this help\n";
}

static EmulatorArgs parse_args(int argc, char *argv[]) {
  EmulatorArgs args;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--device=") == 0)
      args.device = arg.substr(9);
    else if (arg.find("--ib-port=") == 0)
      args.ib_port = std::stoi(arg.substr(10));
    else if (arg.find("--port=") == 0)
      args.control_port = std::stoi(arg.substr(7));
    else if (arg.find("--bridge-ip=") == 0)
      args.bridge_ip = arg.substr(12);
    else if (arg.find("--vp-address=") == 0)
      args.vp_address = std::stoul(arg.substr(13), nullptr, 0);
    else if (arg.find("--hif-address=") == 0)
      args.hif_address = std::stoul(arg.substr(14), nullptr, 0);
    else if (arg.find("--page-size=") == 0)
      args.page_size = std::stoull(arg.substr(12));
    else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      exit(0);
    }
  }
  return args;
}

//==============================================================================
// MAIN
//==============================================================================

int main(int argc, char *argv[]) {
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  try {
    auto args = parse_args(argc, argv);

    std::cout << "=== Hololink FPGA Emulator ===" << std::endl;
    std::cout << "IB Device: " << args.device << std::endl;
    std::cout << "Control port: " << args.control_port << std::endl;
    std::cout << "VP address: 0x" << std::hex << args.vp_address << std::dec
              << std::endl;

    //==========================================================================
    // [1/4] Initialize RDMA
    //==========================================================================
    std::cout << "\n[1/4] Initializing RDMA..." << std::endl;

    RdmaContext rdma;
    if (!rdma.open(args.device, args.ib_port)) {
      std::cerr << "ERROR: Failed to open RDMA device: " << args.device
                << std::endl;
      return 1;
    }
    std::cout << "  GID index: " << rdma.get_gid_index() << std::endl;

    // TX buffer for outgoing syndromes
    RdmaBuffer tx_buffer;
    if (!tx_buffer.allocate(rdma, NUM_BUFFERS * args.page_size)) {
      std::cerr << "ERROR: Failed to allocate TX buffer" << std::endl;
      return 1;
    }

    // RX buffer for incoming responses (same page_size as bridge for
    // symmetry)
    RdmaBuffer rx_buffer;
    if (!rx_buffer.allocate(rdma, NUM_BUFFERS * args.page_size)) {
      std::cerr << "ERROR: Failed to allocate RX buffer" << std::endl;
      return 1;
    }

    // Create CQs and QP
    ibv_cq *tx_cq = rdma.create_cq(NUM_BUFFERS * 2);
    ibv_cq *rx_cq = rdma.create_cq(NUM_BUFFERS * 2);
    if (!tx_cq || !rx_cq) {
      std::cerr << "ERROR: Failed to create CQs" << std::endl;
      return 1;
    }

    ibv_qp *qp = rdma.create_qp(tx_cq, rx_cq, NUM_BUFFERS, NUM_BUFFERS);
    if (!qp) {
      std::cerr << "ERROR: Failed to create QP" << std::endl;
      return 1;
    }
    if (!rdma.qp_to_init(qp)) {
      std::cerr << "ERROR: Failed to set QP to INIT" << std::endl;
      return 1;
    }

    std::cout << "  QP Number: 0x" << std::hex << qp->qp_num << std::dec
              << std::endl;
    std::cout << "  TX buffer: " << tx_buffer.size() << " bytes" << std::endl;
    std::cout << "  RX buffer: " << rx_buffer.size() << " bytes" << std::endl;

    //==========================================================================
    // [2/4] Start UDP control plane server
    //==========================================================================
    std::cout << "\n[2/4] Starting control plane server..." << std::endl;

    RegisterFile regs;
    ControlPlaneServer server(args.control_port, args.vp_address,
                              args.hif_address, regs);
    server.set_my_qp(qp->qp_num);

    if (!server.start()) {
      std::cerr << "ERROR: Failed to start control plane server" << std::endl;
      return 1;
    }
    std::cout << "  Listening on UDP port " << args.control_port << std::endl;
    std::cout << "  Emulator QP: 0x" << std::hex << qp->qp_num << std::dec
              << std::endl;

    //==========================================================================
    // [3/4] Wait for RDMA config from playback tool
    //==========================================================================
    std::cout << "\n[3/4] Waiting for RDMA configuration..." << std::endl;
    std::cout << "  (Start bridge tool, then playback tool with "
                 "--control-port="
              << args.control_port << ")" << std::endl;

    if (!server.wait_for_config(300000)) { // 5 minute timeout
      std::cerr << "ERROR: Timeout waiting for RDMA configuration" << std::endl;
      return 1;
    }

    auto &target = server.target();
    target.print();

    // Connect QP to bridge
    ibv_gid remote_gid{};
    if (!args.bridge_ip.empty()) {
      // Use provided IP
      remote_gid.raw[10] = 0xff;
      remote_gid.raw[11] = 0xff;
      inet_pton(AF_INET, args.bridge_ip.c_str(), &remote_gid.raw[12]);
    } else {
      // Derive from VP HOST_IP register if available
      uint32_t host_ip = regs.read(args.vp_address + 0x28); // DP_HOST_IP
      if (host_ip != 0) {
        remote_gid.raw[10] = 0xff;
        remote_gid.raw[11] = 0xff;
        // DP_HOST_IP is in network byte order from inet_network()
        memcpy(&remote_gid.raw[12], &host_ip, 4);
      } else {
        std::cerr << "ERROR: No bridge IP available. Use --bridge-ip or ensure "
                     "configure_roce sets HOST_IP."
                  << std::endl;
        return 1;
      }
    }

    std::cout << "  Connecting QP to bridge QP 0x" << std::hex
              << target.qp_number << std::dec << "..." << std::endl;

    if (!rdma.qp_to_rtr(qp, remote_gid, target.qp_number, 0)) {
      std::cerr << "ERROR: Failed QP -> RTR" << std::endl;
      return 1;
    }
    if (!rdma.qp_to_rts(qp, 0)) {
      std::cerr << "ERROR: Failed QP -> RTS" << std::endl;
      return 1;
    }
    std::cout << "  QP connected!" << std::endl;

    // Post receive WQEs for responses
    for (size_t i = 0; i < NUM_BUFFERS; i++) {
      void *addr =
          static_cast<uint8_t *>(rx_buffer.data()) + (i * args.page_size);
      if (!rdma.post_recv(qp, i, addr, args.page_size, rx_buffer.lkey())) {
        std::cerr << "ERROR: Failed to post receive WQE " << i << std::endl;
        return 1;
      }
    }
    std::cout << "  Posted " << NUM_BUFFERS << " receive WQEs" << std::endl;

    //==========================================================================
    // [4/4] Wait for playback trigger, then run
    //==========================================================================
    std::cout << "\n[4/4] Waiting for playback trigger..." << std::endl;

    while (!server.playback_triggered() && !g_shutdown) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (g_shutdown) {
      std::cout << "Shutdown requested" << std::endl;
      return 0;
    }

    std::cout << "\n=== Playback triggered ===" << std::endl;

    uint32_t win_size = server.window_size();
    uint32_t win_num = server.window_number();
    uint32_t timer = server.timer_spacing();
    uint32_t cycles_per_window = (win_size + 63) / 64; // 64 bytes per beat

    std::cout << "  Window size: " << win_size << " bytes" << std::endl;
    std::cout << "  Window count: " << win_num << std::endl;
    std::cout << "  Timer spacing: " << timer << " (raw)" << std::endl;
    std::cout << "  Cycles per window: " << cycles_per_window << std::endl;

    // Compute pacing interval from timer register (timer = 322 * microseconds)
    int pacing_us = (timer > 0) ? (timer / 322) : 10;

    // Check if ILA is armed
    bool ila_armed = (regs.read(ILA_CTRL) & 0x01) != 0;
    std::cout << "  ILA capture: " << (ila_armed ? "armed" : "not armed")
              << std::endl;

    // Determine page_size for RDMA addressing from target config
    uint32_t rdma_page_size =
        (target.page_inc > 0) ? target.page_inc : args.page_size;
    uint32_t num_pages = target.max_buff + 1;

    std::cout << "\n=== Starting syndrome transmission ===" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    uint32_t responses_received = 0;
    uint32_t send_errors = 0;
    uint32_t recv_timeouts = 0;

    for (uint32_t window = 0; window < win_num && !g_shutdown; window++) {
      uint32_t slot = window % num_pages;

      // Reassemble syndrome payload from BRAM
      auto payload = reassemble_window(regs, window, cycles_per_window);

      // Copy to RDMA TX buffer slot
      uint8_t *tx_addr =
          static_cast<uint8_t *>(tx_buffer.data()) + (slot * rdma_page_size);
      size_t copy_len = std::min<size_t>(payload.size(), rdma_page_size);
      memcpy(tx_addr, payload.data(), copy_len);

      // RDMA WRITE to bridge's ring buffer
      uint64_t remote_addr = target.buffer_addr + (slot * rdma_page_size);
      if (!rdma.post_rdma_write_imm(qp, window, tx_addr, copy_len,
                                    tx_buffer.lkey(), remote_addr, target.rkey,
                                    slot)) {
        std::cerr << "ERROR: RDMA WRITE failed for window " << window
                  << std::endl;
        send_errors++;
        continue;
      }

      // Wait for send completion
      bool send_ok = false;
      auto t0 = std::chrono::steady_clock::now();
      while (!send_ok && !g_shutdown) {
        ibv_wc wc;
        int n = rdma.poll_cq(tx_cq, &wc, 1);
        if (n > 0) {
          send_ok = (wc.status == IBV_WC_SUCCESS);
          if (!send_ok) {
            std::cerr << "ERROR: Send CQE error: "
                      << ibv_wc_status_str(wc.status) << std::endl;
            send_errors++;
          }
          break;
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        if (elapsed > 5000) {
          std::cerr << "ERROR: Send timeout for window " << window << std::endl;
          recv_timeouts++;
          break;
        }
      }
      if (!send_ok)
        continue;

      // Wait for correction response (natural pacing)
      bool corr_ok = false;
      t0 = std::chrono::steady_clock::now();
      while (!corr_ok && !g_shutdown) {
        ibv_wc wc;
        int n = rdma.poll_cq(rx_cq, &wc, 1);
        if (n > 0) {
          if (wc.status == IBV_WC_SUCCESS) {
            corr_ok = true;
            responses_received++;

            // Store in ILA capture if armed
            if (ila_armed) {
              uint32_t rx_slot = wc.wr_id % NUM_BUFFERS;
              uint8_t *resp_data = static_cast<uint8_t *>(rx_buffer.data()) +
                                   (rx_slot * args.page_size);
              store_ila_sample(regs, window, resp_data, wc.byte_len);
            }

            // Re-post receive WQE
            uint32_t rx_slot = wc.wr_id % NUM_BUFFERS;
            void *rx_addr = static_cast<uint8_t *>(rx_buffer.data()) +
                            (rx_slot * args.page_size);
            rdma.post_recv(qp, rx_slot, rx_addr, args.page_size,
                           rx_buffer.lkey());
          } else {
            std::cerr << "ERROR: Recv CQE error: "
                      << ibv_wc_status_str(wc.status) << std::endl;
          }
          break;
        }
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
        if (elapsed > 10000) {
          std::cerr << "ERROR: Correction timeout for window " << window
                    << std::endl;
          recv_timeouts++;
          break;
        }
      }

      // Progress
      if ((window + 1) % 10 == 0 || window == win_num - 1) {
        std::cout << "  Window " << (window + 1) << "/" << win_num
                  << " (responses: " << responses_received
                  << ", errors: " << send_errors << ")" << std::endl;
      }

      // Pacing delay
      if (pacing_us > 0 && window + 1 < win_num) {
        std::this_thread::sleep_for(std::chrono::microseconds(pacing_us));
      }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_time - start_time);

    // Mark ILA as done
    if (ila_armed) {
      regs.write(ILA_STATUS, regs.read(ILA_STATUS) | 0x02); // done bit
    }

    // Report results
    std::cout << "\n=== Emulator Results ===" << std::endl;
    std::cout << "  Windows sent: " << win_num << std::endl;
    std::cout << "  Responses received: " << responses_received << std::endl;
    std::cout << "  Send errors: " << send_errors << std::endl;
    std::cout << "  Timeouts: " << recv_timeouts << std::endl;
    std::cout << "  Duration: " << duration.count() << " ms" << std::endl;

    // Keep running to allow playback tool to read ILA capture data
    if (ila_armed) {
      std::cout << "\nWaiting for ILA readback (Ctrl+C to stop)..."
                << std::endl;
      while (!g_shutdown) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
    }

    // Cleanup
    server.stop();
    ibv_destroy_qp(qp);
    ibv_destroy_cq(tx_cq);
    ibv_destroy_cq(rx_cq);

    if (send_errors == 0 && recv_timeouts == 0 &&
        responses_received == win_num) {
      std::cout << "\n*** EMULATOR: ALL WINDOWS PROCESSED ***" << std::endl;
      return 0;
    } else {
      std::cout << "\n*** EMULATOR: ERRORS DETECTED ***" << std::endl;
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
