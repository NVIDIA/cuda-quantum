/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "roce_client_common.h"

class RoCEDaemonClient : public RoCEClientBase {
public:
  RoCEDaemonClient(const std::string &device_name)
      : RoCEClientBase(device_name) {
    // Override GID index for DOCA (use RoCEv2 IPv4-mapped GID)
    // GID index 3 = RoCEv2 with IPv4-mapped address (::ffff:192.168.100.1)
    // This matches what DOCAChannel uses on the server side
    gid_index_ = 3;
  }

  ~RoCEDaemonClient() {
    if (qp_)
      ibv_destroy_qp(qp_);
  }

  bool connect_to_server(const std::string &server_ip, uint16_t port) {
    std::cout << "\n[CONNECT] Connecting to server " << server_ip << ":" << port
              << " (UDP)...\n";

    // Create UDP socket
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      std::cerr << "Failed to create UDP socket\n";
      return false;
    }

    struct timeval tv;
    tv.tv_sec = 5;
    tv.tv_usec = 0;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    struct sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);
    inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr);

    // Send DISCOVER
    std::cout << "[UDP] Sending DISCOVER...\n";
    const char *discover = "DISCOVER";
    ssize_t sent = sendto(sock, discover, strlen(discover), 0,
                          (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (sent < 0) {
      std::cerr << "Failed to send DISCOVER\n";
      close(sock);
      return false;
    }

    // Receive channel parameters (single channel for Daemon mode)
    char buffer[4096];
    socklen_t addr_len = sizeof(server_addr);
    ssize_t n = recvfrom(sock, buffer, sizeof(buffer), 0,
                         (struct sockaddr *)&server_addr, &addr_len);
    if (n <= 0) {
      std::cerr << "Failed to receive channel parameters\n";
      close(sock);
      return false;
    }

    try {
      nlohmann::json msg = nlohmann::json::parse(std::string(buffer, n));

      remote_qpn_ = msg["qpn"];
      remote_gid_ = string_to_gid(msg["gid"].get<std::string>());
      server_vaddr_ = msg["vaddr"];
      server_rkey_ = msg["rkey"];
      num_slots_ = msg.value("num_slots", 1024);
      slot_size_ = msg.value("slot_size", 2048);

      std::cout << "[UDP] Received server params:\n";
      std::cout << "  QPN: " << remote_qpn_ << "\n";
      std::cout << "  GID: " << msg["gid"].get<std::string>() << "\n";
      std::cout << "  vaddr: " << server_vaddr_ << "\n";
      std::cout << "  rkey: " << server_rkey_ << "\n";
      std::cout << "  Ring buffer: " << num_slots_ << " slots x " << slot_size_
                << " bytes\n";

    } catch (const std::exception &e) {
      std::cerr << "Failed to parse server message: " << e.what() << "\n";
      close(sock);
      return false;
    }

    // Create QP
    struct ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = send_cq_;
    qp_init_attr.recv_cq = recv_cq_;
    qp_init_attr.qp_type = IBV_QPT_UC;
    qp_init_attr.cap.max_send_wr = 16;
    qp_init_attr.cap.max_recv_wr = 16;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    qp_ = ibv_create_qp(pd_, &qp_init_attr);
    if (!qp_) {
      std::cerr << "Failed to create QP\n";
      close(sock);
      return false;
    }

    local_qpn_ = qp_->qp_num;

    // Send ACK with client params
    nlohmann::json client_params;
    client_params["channel_idx"] = 0; // Single channel for Daemon
    client_params["qpn"] = local_qpn_;
    client_params["gid"] = gid_to_string(local_gid_);

    std::string json_str = client_params.dump();
    sent = sendto(sock, json_str.c_str(), json_str.length(), 0,
                  (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (sent < 0) {
      std::cerr << "Failed to send ACK\n";
      close(sock);
      return false;
    }

    std::cout << "[UDP] Sent ACK with QPN: " << local_qpn_ << "\n";

    // Wait for START message
    n = recvfrom(sock, buffer, sizeof(buffer), 0,
                 (struct sockaddr *)&server_addr, &addr_len);
    if (n > 0) {
      try {
        nlohmann::json msg = nlohmann::json::parse(std::string(buffer, n));
        if (msg.contains("cmd") && msg["cmd"] == "START") {
          std::cout << "[UDP] Received START from server\n";
        }
      } catch (const std::exception &e) {
        // Ignore if not a valid START message
      }
    }

    close(sock);

    // Transition QP to RTS
    std::cout << "\n[QP] Transitioning QP to RTS...\n";
    if (!transition_qp_to_rts(qp_, remote_qpn_, remote_gid_)) {
      return false;
    }
    std::cout << "QP ready (QPN: " << local_qpn_ << ")\n";

    // Post receive buffers
    for (int i = 0; i < 4; i++) {
      if (!post_recv_buffer(qp_)) {
        return false;
      }
    }

    std::cout << "✓ Configuration complete!\n";
    return true;
  }

  bool send_rpc_request(uint32_t function_id, const std::vector<uint8_t> &args,
                        std::vector<uint8_t> &response) {
    // Build RPC request in kernel's expected format (RPCHeader + payload):
    // RPCHeader: [seq_num (8) | function_id (4) | payload_length (4)]
    // Payload:   [args...]
    //
    // This matches the RPCHeader struct in doca_rpc_kernel.cu:
    //   struct RPCHeader { uint64_t sequence_number; uint32_t function_id;
    //   uint32_t payload_length; };
    constexpr size_t RPC_HEADER_SIZE = 16; // sizeof(RPCHeader)
    std::vector<uint8_t> slot_data(RPC_HEADER_SIZE + args.size());
    *(uint64_t *)&slot_data[0] = seq_num_;               // sequence_number
    *(uint32_t *)&slot_data[8] = function_id;            // function_id
    *(uint32_t *)&slot_data[12] = (uint32_t)args.size(); // payload_length
    if (!args.empty()) {
      std::memcpy(&slot_data[16], args.data(), args.size()); // payload
    }

    std::cout << "\n[SEND] RPC Request (function_id=" << function_id
              << ", payload=" << args.size() << " bytes)" << std::endl;

    // Calculate target address in server's buffer
    // NOTE: For DOCA channel, we use direct page-based addressing without a
    // ring header. The kernel expects data at (page * page_size), where page
    // comes from immediate data. This differs from the RoCE ring buffer model
    // which has a 128-byte header.
    uint32_t slot_idx = head_ % num_slots_;
    uint64_t slot_offset =
        slot_idx * slot_size_; // No RING_HEADER_SIZE for DOCA
    uint64_t target_vaddr = server_vaddr_ + slot_offset;

    // Allocate and register local buffer
    void *local_buf = malloc(slot_data.size());
    std::memcpy(local_buf, slot_data.data(), slot_data.size());

    struct ibv_mr *local_mr =
        ibv_reg_mr(pd_, local_buf, slot_data.size(),
                   IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!local_mr) {
      std::cerr << "Failed to register send buffer\n";
      free(local_buf);
      return false;
    }

    // Post RDMA WRITE WITH IMMEDIATE (required for DOCA GPUNetIO)
    // The immediate data generates a CQE that the GPU kernel can poll
    struct ibv_sge sge = {};
    sge.addr = (uint64_t)local_buf;
    sge.length = slot_data.size();
    sge.lkey = local_mr->lkey;

    struct ibv_send_wr wr = {};
    wr.wr_id = send_wr_id_++;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM; // Changed from IBV_WR_RDMA_WRITE
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.rdma.remote_addr = target_vaddr;
    wr.wr.rdma.rkey = server_rkey_;
    // Immediate data: slot index in lower 12 bits (matches Python client and
    // DOCA kernel)
    wr.imm_data = htonl(slot_idx & 0xFFF);

    struct ibv_send_wr *bad_wr = nullptr;
    if (ibv_post_send(qp_, &wr, &bad_wr) != 0) {
      std::cerr << "Failed to post RDMA WRITE WITH IMMEDIATE\n";
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
                << "\n";
      ibv_dereg_mr(local_mr);
      free(local_buf);
      return false;
    }

    std::cout << "[SEND] RDMA WRITE WITH IMMEDIATE sent (seq=" << seq_num_
              << ", slot=" << slot_idx << ", imm_data=" << (slot_idx & 0xFFF)
              << ")\n";

    // Cleanup send buffer
    ibv_dereg_mr(local_mr);
    free(local_buf);

    // Advance ring buffer state
    head_++;
    seq_num_++;

    // Wait for response
    std::cout << "[RECV] Waiting for response...\n";

    ret = 0;
    int poll_count = 0;
    const int max_polls = 5000;

    while (ret == 0 && poll_count < max_polls) {
      ret = ibv_poll_cq(recv_cq_, 1, &wc);
      if (ret == 0) {
        usleep(1000);
        poll_count++;
      }
    }

    if (ret < 0 || poll_count >= max_polls) {
      std::cerr << "[TIMEOUT] No response received\n";
      return false;
    }

    if (wc.status != IBV_WC_SUCCESS) {
      std::cerr << "Receive completion error: " << ibv_wc_status_str(wc.status)
                << "\n";
      return false;
    }

    std::cout << "[RECV] Response received (" << wc.byte_len << " bytes)"
              << "\n";

    // Parse response matching kernel's RPCResponse struct:
    // struct RPCResponse { uint64_t seq; int32_t status; uint32_t result_len;
    // }; Layout: [seq_num (8) | status (4) | result_len (4) | result_data...]
    constexpr size_t RPC_RESPONSE_HEADER_SIZE = 16; // sizeof(RPCResponse)
    if (wc.byte_len < RPC_RESPONSE_HEADER_SIZE) {
      std::cerr << "Response too short: " << wc.byte_len << " bytes\n";
      return false;
    }

    uint8_t *buf = (uint8_t *)recv_buffer_;
    uint64_t resp_seq = *(uint64_t *)&buf[0];
    int32_t status = *(int32_t *)&buf[8];
    uint32_t result_len = *(uint32_t *)&buf[12];

    std::cout << "  Response: seq=" << resp_seq << ", status=" << status
              << ", result_len=" << result_len << "\n";

    if (status != 0) {
      std::cerr << "RPC error: status=" << status << "\n";
      return false;
    }

    response.assign(buf + RPC_RESPONSE_HEADER_SIZE,
                    buf + RPC_RESPONSE_HEADER_SIZE + result_len);

    // Repost receive buffer
    post_recv_buffer(qp_);

    return true;
  }

private:
  struct ibv_qp *qp_ = nullptr;
  uint32_t local_qpn_ = 0;
  uint32_t remote_qpn_ = 0;
  union ibv_gid remote_gid_ = {};
  uint64_t server_vaddr_ = 0;
  uint32_t server_rkey_ = 0;
  uint32_t num_slots_ = 0;
  uint32_t slot_size_ = 0;

  // Ring buffer state
  uint64_t seq_num_ = 1;
  uint32_t head_ = 0;
};

// Test functions
static bool test_add(RoCEDaemonClient &client) {
  std::cout << "\n" << std::string(60, '=') << "\n";
  std::cout << "TEST: Add Function (GPU)\n";
  std::cout << std::string(60, '=') << "\n";

  int32_t a = 42, b = 58;
  int32_t expected = a + b;

  std::vector<uint8_t> args(8);
  *(int32_t *)&args[0] = a;
  *(int32_t *)&args[4] = b;

  std::cout << "  Computing: " << a << " + " << b << " = ?" << "\n";

  std::vector<uint8_t> response;
  // Function ID 2 matches daemon's gpu_add_numbers registration
  if (!client.send_rpc_request(2, args, response)) {
    std::cout << "FAIL: No response\n";
    return false;
  }

  if (response.size() < 4) {
    std::cout << "FAIL: Response too short\n";
    return false;
  }

  int32_t result = *(int32_t *)&response[0];

  std::cout << "  Server returned: " << result << "\n";

  if (result == expected) {
    std::cout << "SUCCESS: " << a << " + " << b << " = " << result
              << " (correct!)" << "\n";
    return true;
  } else {
    std::cout << "FAIL: Expected " << expected << ", got " << result << "\n";
    return false;
  }
}

int main(int argc, char **argv) {
  std::cout << std::string(60, '=') << "\n";
  std::cout << " DOCA Daemon Client (libibverbs)" << "\n";
  std::cout << " Tests GPU RPC function via real RDMA QP" << "\n";
  std::cout << std::string(60, '=') << "\n";

  std::string device_name = "mlx5_1";
  std::string server_ip = "192.168.100.1";
  uint16_t server_port = 9999;

  if (argc > 1)
    device_name = argv[1];
  if (argc > 2)
    server_ip = argv[2];

  std::cout << "\nConfiguration:\n";
  std::cout << "  Device: " << device_name << "\n";
  std::cout << "  Server: " << server_ip << ":" << server_port << "\n\n";

  try {
    RoCEDaemonClient client(device_name);

    if (!client.initialize()) {
      return 1;
    }

    if (!client.connect_to_server(server_ip, server_port)) {
      return 1;
    }

    // Run test
    int success_count = 0;
    int total_tests = 1;

    if (test_add(client)) {
      success_count++;
    }

    // Summary
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "TEST SUMMARY: " << success_count << "/" << total_tests
              << " tests passed" << "\n";
    std::cout << std::string(60, '=') << "\n";

    if (success_count == total_tests) {
      std::cout << "Test passed! DOCA daemon GPU RPC working!\n";
      return 0;
    } else {
      std::cout << "/!\\ Test failed.\n";
      return 1;
    }

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
