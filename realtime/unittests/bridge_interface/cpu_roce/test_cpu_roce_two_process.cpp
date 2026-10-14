/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_cpu_roce_two_process.cpp
/// @brief End-to-end two-process test of the cpu_roce transport: a real
///        dispatcher in a child process serving rpc_increment, driven over
///        RoCE by a caller transceiver in this process.
///
/// The twin of test_udp_two_process: same server, same cases, same shape
/// parameter.  The caller plays CpuRoceChannel's part on the wire -- it
/// Writes-with-Imm requests into the service's rx ring and receives the
/// service's Sends into its own -- after the same TCP {qp, rkey, ip}
/// rendezvous the cpu_roce provider serves in its connect().
///
/// Needs RoCE: the fixture probes the topology from roce_test_topology()
/// (testing/roce_probe.h) and, when it is not usable, skips -- or fails, when
/// RoCE tests are required (build option CUDAQ_REALTIME_REQUIRE_ROCE_TESTS,
/// env CUDAQ_REALTIME_REQUIRE_ROCE).  Both endpoints may name the same port
/// (the HCA loops the frame back internally).
///
/// CONSEQUENCE FOR THE NEGATIVE CASES, as for udp: under the ring shape the
/// service's TX pump walks slots in strict cursor order and parks on a slot
/// the dispatcher never answered, so an undispatchable request must be the
/// LAST one a test sends.  The unified shape publishes by slot index and does
/// not have the problem; the tests are written to the stricter of the two.

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/testing/roce_probe.h"
#include "cudaq/realtime/testing/server_process.h"
#include "cudaq/realtime/testing/test_utils.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifndef CUDAQ_REALTIME_TEST_SERVER_PATH
#error "CUDAQ_REALTIME_TEST_SERVER_PATH must be defined (path to the server)"
#endif

using cudaq::realtime::fnv1a_hash;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;
using cudaq::realtime::testing::roce_test_gate;
using cudaq::realtime::testing::roce_test_topology;
using cudaq::realtime::testing::RoceGate;
using cudaq::realtime::testing::RoceTopology;
using cudaq::realtime::testing::ServerProcess;
using cudaq::realtime::testing::slot_data;
using cudaq::realtime::testing::store_flag;
using cudaq::realtime::testing::wait_for_flag;

namespace {

constexpr const char *kReadyPrefix = "CUDAQ_REALTIME_SERVER_READY";
constexpr const char *kProcessedPrefix = "CUDAQ_REALTIME_SERVER_PROCESSED";
constexpr std::uint32_t kIncrementId = fnv1a_hash("rpc_increment");

constexpr unsigned kNumSlots = 4;
constexpr std::size_t kSlotSize = 256;

constexpr auto kResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kNoResponseWindow = std::chrono::milliseconds(500);

// Must match the provider's RendezvousInfo byte-for-byte (network order).
struct RendezvousInfo {
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t roce_ipv4 = 0;
};

bool write_all(int fd, const void *buf, std::size_t len) {
  const auto *p = static_cast<const std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::write(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

bool read_all(int fd, void *buf, std::size_t len) {
  auto *p = static_cast<std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::read(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

// The client half of the provider's rendezvous: send ours, read theirs.
bool exchange_rendezvous(std::uint16_t port, const RendezvousInfo &self,
                         RendezvousInfo &peer) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0)
    return false;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  int one = 1;
  ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
  const bool ok =
      ::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0 &&
      write_all(fd, &self, sizeof(self)) && read_all(fd, &peer, sizeof(peer));
  ::close(fd);
  return ok;
}

class CpuRoceTwoProcess : public ::testing::TestWithParam<const char *> {
protected:
  void SetUp() override {
    topology = roce_test_topology();
    const RoceGate gate = roce_test_gate(topology);
    if (gate.action == RoceGate::Fail)
      FAIL() << gate.reason;
    if (gate.action == RoceGate::Skip)
      GTEST_SKIP() << gate.reason;
    const char *caller_dev = topology.caller.device.c_str();
    const char *caller_ip = topology.caller.ip.c_str();

    const bool unified = std::string(GetParam()) == "unified";
    std::vector<std::string> argv = {
        CUDAQ_REALTIME_TEST_SERVER_PATH, "--transport=cpu_roce",
        std::string("--dispatch=") + GetParam(), "--timeout=120",
        // Bridge options from here on, forwarded verbatim by the server.
        "--", "--device=" + topology.service.device,
        "--local-ip=" + topology.service.ip, "--port=0",
        "--num-slots=" + std::to_string(kNumSlots),
        "--slot-size=" + std::to_string(kSlotSize)};
    if (unified)
      argv.push_back("--unified");
    ASSERT_TRUE(server.start(argv, kReadyPrefix))
        << "server did not become ready; output:\n"
        << server.output();
    EXPECT_EQ(std::string(GetParam()), field("dispatch")) << server.output();
    ASSERT_EQ("cpu_roce", field("transport")) << server.output();
    ASSERT_EQ(std::to_string(kNumSlots), field("slots"));
    ASSERT_EQ(std::to_string(kSlotSize), field("slot_size"));
    ASSERT_NE(0, server.port()) << server.output();

    caller = cpu_roce_create_transceiver(
        caller_dev, 1, 0, kSlotSize, kSlotSize, kNumSlots, "0.0.0.0", 0, 0, 0,
        0, CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM, 0, 0);
    ASSERT_NE(nullptr, caller);
    cpu_roce_set_local_ip(caller, caller_ip);
    ASSERT_EQ(1, cpu_roce_setup(caller));

    in_addr local{};
    ASSERT_EQ(1, ::inet_pton(AF_INET, caller_ip, &local));
    const RendezvousInfo self{htonl(cpu_roce_get_qp_number(caller)),
                              htonl(cpu_roce_get_rkey(caller)), local.s_addr};
    RendezvousInfo peer{};
    ASSERT_TRUE(exchange_rendezvous(static_cast<std::uint16_t>(server.port()),
                                    self, peer))
        << server.output();
    char peer_ip[INET_ADDRSTRLEN] = {0};
    in_addr peer_addr{};
    peer_addr.s_addr = peer.roce_ipv4;
    ASSERT_NE(nullptr,
              ::inet_ntop(AF_INET, &peer_addr, peer_ip, sizeof(peer_ip)));
    ASSERT_EQ(1, cpu_roce_connect(caller, ntohl(peer.qp_number), peer_ip,
                                  ntohl(peer.rkey)));
    callerMonitor = std::thread([this] { cpu_roce_blocking_monitor(caller); });

    // UC drops a frame that reaches a QP not yet at RTR, without telling
    // either side, and the service moves to RTR only after replying to the
    // rendezvous.  Give it that transition before the first request.
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    txFlags =
        reinterpret_cast<std::uint64_t>(cpu_roce_get_tx_ring_flag_addr(caller));
    txData =
        reinterpret_cast<std::uint64_t>(cpu_roce_get_tx_ring_data_addr(caller));
    rxFlags =
        reinterpret_cast<std::uint64_t>(cpu_roce_get_rx_ring_flag_addr(caller));
    rxData =
        reinterpret_cast<std::uint64_t>(cpu_roce_get_rx_ring_data_addr(caller));
  }

  void TearDown() override {
    if (caller)
      cpu_roce_close(caller);
    if (callerMonitor.joinable())
      callerMonitor.join();
    if (caller)
      cpu_roce_destroy_transceiver(caller);
    server.stop();
  }

  std::string field(const std::string &key) const {
    const auto it = server.fields().find(key);
    return it == server.fields().end() ? std::string{} : it->second;
  }

  // Publish one request from the next TX slot; the caller's TX pump Writes it
  // into the same-index slot of the service's rx ring.
  void postIncrement(std::uint32_t request_id, std::string_view args,
                     std::uint32_t magic = RPC_MAGIC_REQUEST,
                     std::uint32_t function_id = kIncrementId) {
    ASSERT_LE(sizeof(RPCHeader) + args.size(), kSlotSize);
    std::uint8_t *tx = slot_data(txData, txCursor, kSlotSize);
    std::memset(tx, 0, kSlotSize);

    RPCHeader header{};
    header.magic = magic;
    header.function_id = function_id;
    header.arg_len = static_cast<std::uint32_t>(args.size());
    header.request_id = request_id;
    header.ptp_timestamp = 0;
    std::memcpy(tx, &header, sizeof(header));
    if (!args.empty())
      std::memcpy(tx + sizeof(header), args.data(), args.size());

    store_flag(txFlags, txCursor, reinterpret_cast<std::uint64_t>(tx));
    txCursor = (txCursor + 1) % kNumSlots;
  }

  void expectIncrement(std::uint32_t request_id, std::string_view sent) {
    const unsigned slot = rxCursor;
    ASSERT_TRUE(wait_for_flag(rxFlags, slot, kResponseTimeout))
        << "no response in rx slot " << slot << " for request " << request_id
        << "; server output:\n"
        << server.output();

    const std::uint8_t *rx = slot_data(rxData, slot, kSlotSize);
    RPCResponse response{};
    std::memcpy(&response, rx, sizeof(response));
    EXPECT_EQ(RPC_MAGIC_RESPONSE, response.magic);
    EXPECT_EQ(0, response.status);
    EXPECT_EQ(static_cast<std::uint32_t>(sent.size()), response.result_len);
    EXPECT_EQ(request_id, response.request_id);

    std::string expected(sent.size(), '\0');
    for (std::size_t i = 0; i < sent.size(); ++i)
      expected[i] = static_cast<char>(sent[i] + 1);
    const auto *payload = reinterpret_cast<const char *>(rx + sizeof(response));
    EXPECT_EQ(expected, std::string(payload, sent.size()));

    store_flag(rxFlags, slot, 0);
    rxCursor = (rxCursor + 1) % kNumSlots;
  }

  void expectNoResponse() {
    EXPECT_FALSE(wait_for_flag(rxFlags, rxCursor, kNoResponseWindow))
        << "expected no response in rx slot " << rxCursor
        << "; server output:\n"
        << server.output();
  }

  void expectProcessedCount(unsigned long long expected) {
    const std::string line =
        server.stopAndReadLine(kProcessedPrefix, std::chrono::seconds(10));
    ASSERT_FALSE(line.empty()) << "no processed-count line; server output:\n"
                               << server.output();
    unsigned long long count = 0;
    ASSERT_EQ(1,
              std::sscanf(line.c_str(),
                          "CUDAQ_REALTIME_SERVER_PROCESSED count=%llu", &count))
        << "unparsable line: " << line;
    EXPECT_EQ(expected, count);
  }

  // A member, not a SetUp() local: the transceiver keeps the device-name
  // pointer it is created with.
  RoceTopology topology;
  ServerProcess server;
  cpu_roce_transceiver_t caller = nullptr;
  std::thread callerMonitor;
  std::uint64_t txFlags = 0, txData = 0, rxFlags = 0, rxData = 0;
  unsigned txCursor = 0, rxCursor = 0;
};

TEST_P(CpuRoceTwoProcess, IncrementsStringPayload) {
  const std::string payload = "hello dispatcher";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);
}

TEST_P(CpuRoceTwoProcess, IncrementsBurstAcrossRingWrap) {
  // Three times around both rings, paced one request at a time (the v1
  // contract): a slot whose recv WQE was not re-armed stalls the run.
  const unsigned requests = 3 * kNumSlots;
  for (unsigned i = 0; i < requests; ++i) {
    const std::string payload = "burst-" + std::to_string(i);
    postIncrement(/*request_id=*/i + 1, payload);
    expectIncrement(/*request_id=*/i + 1, payload);
  }
}

TEST_P(CpuRoceTwoProcess, DropsBadMagic) {
  const std::string payload = "before bad magic";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);

  postIncrement(/*request_id=*/2, "unframed", /*magic=*/0xdeadbeefu);
  expectNoResponse();

  expectProcessedCount(1);
}

TEST_P(CpuRoceTwoProcess, DropsUnknownFunctionId) {
  const std::string payload = "before unknown id";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);

  postIncrement(/*request_id=*/2, "unroutable", RPC_MAGIC_REQUEST,
                /*function_id=*/fnv1a_hash("no_such_function"));
  expectNoResponse();

  expectProcessedCount(1);
}

TEST_P(CpuRoceTwoProcess, ReportsProcessedCount) {
  constexpr unsigned kRequests = 5;
  for (unsigned i = 0; i < kRequests; ++i) {
    const std::string payload = "counted-" + std::to_string(i);
    postIncrement(/*request_id=*/i + 1, payload);
    expectIncrement(/*request_id=*/i + 1, payload);
  }
  expectProcessedCount(kRequests);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, CpuRoceTwoProcess, ::testing::Values("ring", "unified"),
    [](const ::testing::TestParamInfo<const char *> &info) {
      return std::string(info.param);
    });

} // namespace
