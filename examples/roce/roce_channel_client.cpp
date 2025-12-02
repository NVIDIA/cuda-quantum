/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <arpa/inet.h>
#include <cstring>
#include <infiniband/verbs.h>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

// RPC Function IDs (for routing to correct channel)
constexpr uint32_t FUNCTION_ECHO = 1;
constexpr uint32_t FUNCTION_ADD = 2;
constexpr uint32_t FUNCTION_MULTIPLY = 3;

// Ring buffer layout (must match server)
constexpr size_t RING_HEADER_SIZE = 128;
constexpr size_t SLOT_HEADER_SIZE = 16;

/**
 * RoCE Full Client - Two-way RDMA communication using libibverbs
 *
 * This client properly creates QPs and CQs to send RDMA WRITE requests
 * and receive RDMA SEND responses from the server.
 */
class RoCEFullClient {
public:
  RoCEFullClient(const std::string &device_name) : device_name_(device_name) {}

  ~RoCEFullClient() { cleanup(); }

  bool initialize() {
    std::cout << "\n[INIT] Initializing InfiniBand Verbs..." << std::endl;

    // Get device list
    int num_devices;
    dev_list_ = ibv_get_device_list(&num_devices);
    if (!dev_list_ || num_devices == 0) {
      std::cerr << "No InfiniBand devices found" << std::endl;
      return false;
    }

    // Find the specified device
    ibv_device *device = nullptr;
    for (int i = 0; i < num_devices; i++) {
      if (device_name_ == ibv_get_device_name(dev_list_[i])) {
        device = dev_list_[i];
        break;
      }
    }

    if (!device) {
      std::cerr << "Device " << device_name_ << " not found" << std::endl;
      return false;
    }

    // Open device
    context_ = ibv_open_device(device);
    if (!context_) {
      std::cerr << "Failed to open device" << std::endl;
      return false;
    }

    // Query port attributes
    if (ibv_query_port(context_, port_num_, &port_attr_) != 0) {
      std::cerr << "Failed to query port" << std::endl;
      return false;
    }

    // Allocate protection domain
    pd_ = ibv_alloc_pd(context_);
    if (!pd_) {
      std::cerr << "Failed to allocate protection domain" << std::endl;
      return false;
    }

    // Create completion queues
    send_cq_ = ibv_create_cq(context_, 16, nullptr, nullptr, 0);
    recv_cq_ = ibv_create_cq(context_, 16, nullptr, nullptr, 0);
    if (!send_cq_ || !recv_cq_) {
      std::cerr << "Failed to create completion queues" << std::endl;
      return false;
    }

    // Note: QPs will be created dynamically per channel during connection

    // Get local GID
    if (ibv_query_gid(context_, port_num_, gid_index_, &local_gid_) != 0) {
      std::cerr << "Failed to query GID" << std::endl;
      return false;
    }

    // Allocate receive buffers
    recv_buffer_size_ = 4096;
    recv_buffer_ = malloc(recv_buffer_size_);
    if (!recv_buffer_) {
      std::cerr << "Failed to allocate receive buffer" << std::endl;
      return false;
    }

    recv_mr_ = ibv_reg_mr(pd_, recv_buffer_, recv_buffer_size_,
                          IBV_ACCESS_LOCAL_WRITE);
    if (!recv_mr_) {
      std::cerr << "Failed to register receive buffer" << std::endl;
      return false;
    }

    std::cout << "✓ InfiniBand Verbs initialized" << std::endl;
    std::cout << "  Device: " << device_name_ << std::endl;
    std::cout << "  Local GID: " << gid_to_string(local_gid_) << std::endl;

    return true;
  }

  bool connect_to_server(const std::string &server_ip, uint16_t port) {
    std::cout << "\n[CONNECT] Connecting to server " << server_ip << ":" << port
              << " (UDP)..." << std::endl;

    // Create UDP socket for control channel
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      std::cerr << "Failed to create UDP socket" << std::endl;
      return false;
    }

    // Set receive timeout
    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

    // Step 1: Send DISCOVER to announce presence
    std::cout << "[UDP] Sending DISCOVER..." << std::endl;
    const char *discover = "DISCOVER";
    ssize_t sent = sendto(sock, discover, strlen(discover), 0,
                          (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (sent < 0) {
      std::cerr << "Failed to send DISCOVER" << std::endl;
      close(sock);
      return false;
    }

    // Step 2: Stay in configuration state until server sends START
    char buffer[4096];
    socklen_t addr_len = sizeof(server_addr);
    uint32_t channels_configured = 0;
    bool start_received = false;

    std::cout << "\n[UDP] === Configuration State ===" << std::endl;
    std::cout << "Waiting for server to send channel parameters..."
              << std::endl;

    while (!start_received) {
      // Receive message from server
      ssize_t n = recvfrom(sock, buffer, sizeof(buffer), 0,
                           (struct sockaddr *)&server_addr, &addr_len);
      if (n <= 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
          std::cerr << "[UDP] Timeout waiting for server" << std::endl;
          close(sock);
          return false;
        }
        std::cerr << "Failed to receive from server" << std::endl;
        close(sock);
        return false;
      }

      try {
        nlohmann::json msg = nlohmann::json::parse(std::string(buffer, n));

        // Check if this is a START message
        if (msg.contains("cmd") && msg["cmd"] == "START") {
          std::cout << "\n[UDP] ✅ Received START from server" << std::endl;
          uint32_t num_channels = msg.value("num_channels", 0);
          std::cout << "  Server has " << num_channels << " channel(s)"
                    << std::endl;
          std::cout << "  Client configured " << channels_configured
                    << " channel(s)" << std::endl;

          // Parse function routing
          if (msg.contains("function_routing")) {
            auto routing = msg["function_routing"];
            std::cout << "  Function routing:" << std::endl;
            for (auto &[func_id_str, chan_idx] : routing.items()) {
              uint32_t func_id = std::stoi(func_id_str);
              uint32_t channel_idx = chan_idx;
              function_routing_[func_id] = channel_idx;
              std::cout << "    Function " << func_id << " → channel "
                        << channel_idx << std::endl;
            }
          }

          start_received = true;
          break;
        }

        // Otherwise, this is a channel params message
        if (!msg.contains("channel_idx")) {
          std::cerr << "Warning: Received message without channel_idx"
                    << std::endl;
          continue;
        }

        uint32_t channel_idx = msg["channel_idx"];
        std::cout << "\n[UDP] === Channel " << channel_idx
                  << " exchange ===" << std::endl;

        // Create a new channel entry
        ChannelParams channel;

        // Create QP for this channel
        struct ibv_qp_init_attr qp_init_attr = {};
        qp_init_attr.send_cq = send_cq_;
        qp_init_attr.recv_cq = recv_cq_;
        qp_init_attr.qp_type = IBV_QPT_UC;
        qp_init_attr.cap.max_send_wr = 16;
        qp_init_attr.cap.max_recv_wr = 16;
        qp_init_attr.cap.max_send_sge = 1;
        qp_init_attr.cap.max_recv_sge = 1;

        channel.qp = ibv_create_qp(pd_, &qp_init_attr);
        if (!channel.qp) {
          std::cerr << "Failed to create QP for channel " << channel_idx
                    << std::endl;
          close(sock);
          return false;
        }

        channel.local_qpn = channel.qp->qp_num;

        // Parse server params
        channel.remote_qpn = msg["qpn"];
        channel.remote_gid = string_to_gid(msg["gid"].get<std::string>());
        channel.server_vaddr = msg["vaddr"];
        channel.server_rkey = msg["rkey"];
        channel.num_slots = msg.value("num_slots", 1024);
        channel.slot_size = msg.value("slot_size", 2048);

        std::cout << "  Server → Client:" << std::endl;
        std::cout << "    QPN: " << channel.remote_qpn << std::endl;
        std::cout << "    GID: " << msg["gid"].get<std::string>() << std::endl;
        std::cout << "    Ring buffer: " << channel.num_slots << " slots x "
                  << channel.slot_size << " bytes" << std::endl;

        // Send ACK with client params
        nlohmann::json client_params;
        client_params["channel_idx"] = channel_idx;
        client_params["qpn"] = channel.local_qpn;
        client_params["gid"] = gid_to_string(local_gid_);

        std::string json_str = client_params.dump();
        sent = sendto(sock, json_str.c_str(), json_str.length(), 0,
                      (struct sockaddr *)&server_addr, sizeof(server_addr));
        if (sent < 0) {
          std::cerr << "Failed to send ACK" << std::endl;
          close(sock);
          return false;
        }

        std::cout << "  Client → Server: ACK sent" << std::endl;
        std::cout << "    QPN: " << channel.local_qpn << std::endl;
        std::cout << "    GID: " << gid_to_string(local_gid_) << std::endl;

        channels_.push_back(channel);
        channels_configured++;

      } catch (const std::exception &e) {
        std::cerr << "Failed to parse server message: " << e.what()
                  << std::endl;
        close(sock);
        return false;
      }
    }

    if (!start_received) {
      std::cerr << "Failed to receive START from server" << std::endl;
      close(sock);
      return false;
    }

    close(sock);

    std::cout << "\n[QP] Transitioning all " << channels_.size()
              << " QPs to RTS..." << std::endl;

    // Transition all QPs to RTS
    for (size_t i = 0; i < channels_.size(); i++) {
      std::cout << "  Channel " << i << ": QPN " << channels_[i].local_qpn
                << " → RTS..." << std::endl;
      if (!transition_qp_to_rts(i)) {
        return false;
      }
      std::cout << "    ✓ Channel " << i << " ready" << std::endl;
    }

    // Post receive buffers for RDMA SEND responses
    if (!post_recv_buffers(4)) {
      return false;
    }

    std::cout << "\n✓ Configuration complete! Transitioning to RPC state..."
              << std::endl;
    return true;
  }

  bool send_rpc_request(uint32_t function_id, const std::vector<uint8_t> &args,
                        std::vector<uint8_t> &response) {
    // Route to correct channel based on function_id
    auto it = function_routing_.find(function_id);
    if (it == function_routing_.end()) {
      std::cerr << "No routing for function_id " << function_id << std::endl;
      return false;
    }

    uint32_t channel_idx = it->second;
    if (channel_idx >= channels_.size()) {
      std::cerr << "Channel index " << channel_idx << " out of range (have "
                << channels_.size() << " channels)" << std::endl;
      return false;
    }

    auto &ch = channels_[channel_idx];

    std::cout << "  [ROUTE] Function " << function_id << " → channel "
              << channel_idx << std::endl;

    // Build payload: [arg_len | args]
    // No function_id in payload - queue selection IS the routing!
    std::vector<uint8_t> payload(4 + args.size());
    *(uint32_t *)&payload[0] = (uint32_t)args.size();
    if (!args.empty()) {
      std::memcpy(&payload[4], args.data(), args.size());
    }

    std::cout << "\n[SEND] RPC Request to channel " << channel_idx
              << " (payload=" << payload.size() << " bytes)" << std::endl;

    // Build slot data for ring buffer
    std::vector<uint8_t> slot_data(SLOT_HEADER_SIZE + payload.size());
    *(uint64_t *)&slot_data[0] = ch.seq_num; // seq_num (per-channel)
    *(uint32_t *)&slot_data[8] = (uint32_t)payload.size(); // payload_len
    *(uint32_t *)&slot_data[12] = 0;                       // reserved
    std::memcpy(&slot_data[16], payload.data(), payload.size());

    // Calculate target address in server's ring buffer (per-channel head)
    uint32_t slot_idx = ch.head % ch.num_slots;
    uint64_t slot_offset = RING_HEADER_SIZE + (slot_idx * ch.slot_size);
    uint64_t target_vaddr = ch.server_vaddr + slot_offset;

    // We need to allocate local memory for the slot data
    void *local_buf = malloc(slot_data.size());
    std::memcpy(local_buf, slot_data.data(), slot_data.size());

    struct ibv_mr *local_mr =
        ibv_reg_mr(pd_, local_buf, slot_data.size(),
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!local_mr) {
      std::cerr << "Failed to register send buffer" << std::endl;
      free(local_buf);
      return false;
    }

    // Post RDMA WRITE
    struct ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = slot_data.size();
    sge.lkey = local_mr->lkey;

    struct ibv_send_wr wr = {};
    wr.wr_id = send_wr_id_++;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = target_vaddr;
    wr.wr.rdma.rkey = ch.server_rkey;

    struct ibv_send_wr *bad_wr = nullptr;
    if (ibv_post_send(ch.qp, &wr, &bad_wr) != 0) {
      std::cerr << "Failed to post RDMA WRITE" << std::endl;
      ibv_dereg_mr(local_mr);
      free(local_buf);
      return false;
    }

    // Poll for send completion
    struct ibv_wc wc;
    int ret;
    do {
      ret = ibv_poll_cq(send_cq_, 1, &wc);
    } while (ret == 0);

    if (ret < 0 || wc.status != IBV_WC_SUCCESS) {
      std::cerr << "Send completion error: " << ibv_wc_status_str(wc.status)
                << std::endl;
      ibv_dereg_mr(local_mr);
      free(local_buf);
      return false;
    }

    std::cout << "[SEND] ✓ RDMA WRITE sent (seq=" << ch.seq_num
              << ", slot=" << slot_idx << ", channel=" << channel_idx << ")"
              << std::endl;

    // Cleanup send buffer
    ibv_dereg_mr(local_mr);
    free(local_buf);

    // Advance ring buffer state (per-channel)
    ch.head++;
    ch.seq_num++;

    // Wait for response (RDMA SEND from server)
    std::cout << "[RECV] Waiting for response..." << std::endl;

    ret = 0;
    int poll_count = 0;
    const int max_polls = 5000; // 5 second timeout (1ms per poll)

    while (ret == 0 && poll_count < max_polls) {
      ret = ibv_poll_cq(recv_cq_, 1, &wc);
      if (ret == 0) {
        usleep(1000); // 1ms
        poll_count++;
      }
    }

    if (ret < 0 || poll_count >= max_polls) {
      std::cerr << "[TIMEOUT] No response received" << std::endl;
      return false;
    }

    if (wc.status != IBV_WC_SUCCESS) {
      std::cerr << "Receive completion error: " << ibv_wc_status_str(wc.status)
                << std::endl;
      return false;
    }

    std::cout << "[RECV] ✓ Response received (" << wc.byte_len << " bytes)"
              << std::endl;

    // Parse response
    response.assign((uint8_t *)recv_buffer_,
                    (uint8_t *)recv_buffer_ + wc.byte_len);

    // Repost receive buffer to the channel that just received
    // (In this simple client, we just repost to all - could be optimized)
    post_recv_buffers(1);

    return true;
  }

private:
  void cleanup() {
    for (auto &ch : channels_) {
      if (ch.qp)
        ibv_destroy_qp(ch.qp);
    }
    channels_.clear();
    if (send_cq_)
      ibv_destroy_cq(send_cq_);
    if (recv_cq_)
      ibv_destroy_cq(recv_cq_);
    if (recv_mr_)
      ibv_dereg_mr(recv_mr_);
    if (recv_buffer_)
      free(recv_buffer_);
    if (pd_)
      ibv_dealloc_pd(pd_);
    if (context_)
      ibv_close_device(context_);
    if (dev_list_)
      ibv_free_device_list(dev_list_);
  }

  bool transition_qp_to_rts(size_t channel_idx) {
    if (channel_idx >= channels_.size()) {
      std::cerr << "Invalid channel index: " << channel_idx << std::endl;
      return false;
    }

    auto &ch = channels_[channel_idx];

    // RESET -> INIT
    struct ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = port_num_;
    attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    if (ibv_modify_qp(ch.qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to INIT" << std::endl;
      return false;
    }

    // INIT -> RTR
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_1024;
    attr.dest_qp_num = ch.remote_qpn;
    attr.rq_psn = 0;

    // Address vector
    attr.ah_attr.dlid = 0;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = port_num_;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid = ch.remote_gid;
    attr.ah_attr.grh.sgid_index = gid_index_;
    attr.ah_attr.grh.hop_limit = 64;

    mask = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
           IBV_QP_RQ_PSN;
    if (ibv_modify_qp(ch.qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to RTR" << std::endl;
      return false;
    }

    // RTR -> RTS
    std::memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;

    mask = IBV_QP_STATE | IBV_QP_SQ_PSN;
    if (ibv_modify_qp(ch.qp, &attr, mask) != 0) {
      std::cerr << "Failed to transition QP to RTS" << std::endl;
      return false;
    }

    return true;
  }

  bool post_recv_buffers(int count) {
    // Post recv buffers to ALL QPs (responses can come from any channel)
    for (auto &ch : channels_) {
      for (int i = 0; i < count; i++) {
        struct ibv_sge sge = {};
        sge.addr = (uint64_t)recv_buffer_;
        sge.length = recv_buffer_size_;
        sge.lkey = recv_mr_->lkey;

        struct ibv_recv_wr wr = {};
        wr.wr_id = recv_wr_id_++;
        wr.sg_list = &sge;
        wr.num_sge = 1;

        struct ibv_recv_wr *bad_wr = nullptr;
        if (ibv_post_recv(ch.qp, &wr, &bad_wr) != 0) {
          std::cerr << "Failed to post receive buffer to QP " << ch.local_qpn
                    << std::endl;
          return false;
        }
      }
    }
    return true;
  }

  std::string gid_to_string(const union ibv_gid &gid) {
    char buf[64];
    inet_ntop(AF_INET6, gid.raw, buf, sizeof(buf));
    return buf;
  }

  union ibv_gid string_to_gid(const std::string &gid_str) {
    union ibv_gid gid;
    inet_pton(AF_INET6, gid_str.c_str(), gid.raw);
    return gid;
  }

  std::string device_name_;
  uint8_t port_num_ = 1;
  uint8_t gid_index_ = 1; // Use GID index 1 for Soft-RoCE

  // IB Verbs resources
  struct ibv_device **dev_list_ = nullptr;
  struct ibv_context *context_ = nullptr;
  struct ibv_pd *pd_ = nullptr;
  struct ibv_cq *send_cq_ = nullptr;
  struct ibv_cq *recv_cq_ = nullptr;
  struct ibv_qp *qp_ = nullptr;
  struct ibv_port_attr port_attr_ = {};

  // Connection params per channel
  struct ChannelParams {
    uint32_t local_qpn;
    uint32_t remote_qpn;
    union ibv_gid remote_gid;
    uint64_t server_vaddr;
    uint32_t server_rkey;
    uint32_t num_slots;
    uint32_t slot_size;
    struct ibv_qp *qp;

    // Per-channel ring buffer state
    uint64_t seq_num = 1;
    uint32_t head = 0;
  };

  std::vector<ChannelParams> channels_;
  union ibv_gid local_gid_ = {};
  std::map<uint32_t, uint32_t> function_routing_; // function_id → channel_idx

  // Receive buffers
  void *recv_buffer_ = nullptr;
  size_t recv_buffer_size_ = 0;
  struct ibv_mr *recv_mr_ = nullptr;

  // Ring buffer state is now per-channel (in ChannelParams)

  // Work request IDs
  uint64_t send_wr_id_ = 0;
  uint64_t recv_wr_id_ = 0;
};

// Test functions
static bool test_echo(RoCEFullClient &client) {
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TEST 1: Echo Function" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  std::string message = "Hello from RoCE full client!";
  std::vector<uint8_t> args(message.begin(), message.end());
  std::vector<uint8_t> response;

  if (!client.send_rpc_request(FUNCTION_ECHO, args, response)) {
    std::cout << "❌ FAIL: No response" << std::endl;
    return false;
  }

  // Parse response: [result_data] (Channel mode - no RPC headers)
  std::string result(response.begin(), response.end());

  if (result == message) {
    std::cout << "✅ SUCCESS: Echo returned correct message!" << std::endl;
    std::cout << "  Sent: '" << message << "'" << std::endl;
    std::cout << "  Received: '" << result << "'" << std::endl;
    return true;
  } else {
    std::cout << "❌ FAIL: Sent '" << message << "', got '" << result << "'"
              << std::endl;
    return false;
  }
}

static bool test_add(RoCEFullClient &client) {
  std::cout << "\n" << std::string(60, '=') << std::endl;
  std::cout << "TEST 2: Add Function" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  int32_t a = 42, b = 58;
  int32_t expected = a + b;

  std::vector<uint8_t> args(8);
  *(int32_t *)&args[0] = a;
  *(int32_t *)&args[4] = b;

  std::cout << "  Computing: " << a << " + " << b << " = ?" << std::endl;

  std::vector<uint8_t> response;
  if (!client.send_rpc_request(FUNCTION_ADD, args, response)) {
    std::cout << "❌ FAIL: No response" << std::endl;
    return false;
  }

  if (response.size() < 4) {
    std::cout << "❌ FAIL: Response too short (got " << response.size()
              << " bytes, expected 4)" << std::endl;
    return false;
  }

  // Parse response: [result] (Channel mode - no RPC headers)
  int32_t result = *(int32_t *)&response[0];

  std::cout << "  Server returned: " << result << std::endl;

  if (result == expected) {
    std::cout << "✅ SUCCESS: " << a << " + " << b << " = " << result
              << " (correct!)" << std::endl;
    return true;
  } else {
    std::cout << "❌ FAIL: Expected " << expected << ", got " << result
              << std::endl;
    return false;
  }
}

int main(int argc, char **argv) {
  std::cout << std::string(60, '=') << std::endl;
  std::cout << " RoCEv2 Full Client (Linux + libibverbs)" << std::endl;
  std::cout << " Two-way RDMA communication with proper QP setup" << std::endl;
  std::cout << std::string(60, '=') << std::endl;

  std::string device_name = "rxe0";
  std::string server_ip = "127.0.0.1";
  uint16_t server_port = 9999;

  if (argc > 1)
    device_name = argv[1];
  if (argc > 2)
    server_ip = argv[2];

  std::cout << "\nConfiguration:" << std::endl;
  std::cout << "  Device: " << device_name << std::endl;
  std::cout << "  Server: " << server_ip << ":" << server_port << std::endl;
  std::cout << std::endl;

  try {
    RoCEFullClient client(device_name);

    if (!client.initialize()) {
      return 1;
    }

    if (!client.connect_to_server(server_ip, server_port)) {
      return 1;
    }

    // Run tests
    int success_count = 0;
    int total_tests = 2;

    if (test_echo(client)) {
      success_count++;
    }

    sleep(1);

    if (test_add(client)) {
      success_count++;
    }

    // Summary
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "TEST SUMMARY: " << success_count << "/" << total_tests
              << " tests passed" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    if (success_count == total_tests) {
      std::cout << "🎉 All tests passed! RoCEv2 two-way communication working!"
                << std::endl;
      return 0;
    } else {
      std::cout << "⚠️  Some tests failed." << std::endl;
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "\n❌ Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
