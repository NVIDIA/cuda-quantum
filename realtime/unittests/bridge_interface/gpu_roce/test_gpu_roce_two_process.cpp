/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_gpu_roce_two_process.cpp
/// @brief End-to-end two-process test of the gpu_roce transport: a DEVICE
///        dispatcher in a child process serving rpc_increment on the GPU,
///        driven over RoCE by a cpu_roce caller in this process.
///
/// The twin of test_cpu_roce_two_process: same server (with --path=device),
/// same cases, same shape parameter.  `ring` is the 3-kernel arrangement (the
/// provider's RX and TX kernels around the dispatch kernel), `unified` the
/// single kernel that drives the provider's device hooks.  The caller plays
/// the FPGA's part on the wire: it Writes-with-Imm each request into the
/// service's rx ring (registered at IOVA 0, so slot N is at N * stride, with N
/// in the immediate) and receives the service's Sends into its own.
///
/// NO RENDEZVOUS.  gpu_roce takes its peer when it is created and its
/// connect() only prints, so the order is fixed by who learns what first:
///   1. the caller builds its QP (cpu_roce_setup) to learn its QP number;
///   2. it spawns the server with that number and its own IP as the peer;
///   3. it reads the service's QP and rkey from the READY line;
///   4. it moves its QP to RTS against them (cpu_roce_connect).
///
/// Needs RoCE and a GPU: the fixture probes the topology from
/// roce_test_topology() (testing/roce_probe.h) and, when it is not usable,
/// skips -- or fails, when RoCE tests are required (build option
/// CUDAQ_REALTIME_REQUIRE_ROCE_TESTS, env CUDAQ_REALTIME_REQUIRE_ROCE).  The
/// service is the DAEMON endpoint, the caller the CHANNEL one; both may name
/// the same port (the HCA loops the frame back internally).
///
/// THE SHAPES DIFFER ON AN UNDISPATCHABLE FRAME, and the negative cases expect
/// each one's behavior through expectErrorResponse():
///   - ring: the dispatch kernel drops it -- no response, not counted.  The
///     provider's TX kernel walks slots in strict order and parks on the one
///     never answered, so such a request must be the LAST one a test sends.
///   - unified: the kernel answers every frame, with status -1 and no result,
///     and counts it.
///
/// KNOWN ISSUE, ring shape only: the server reports its count but never exits
/// on its own, because HSB's GpuRoceTransceiver destructor waits on an RX
/// kernel that does not end after close() (gpu_roce_app hangs the same way).
/// ServerProcess kills it after its grace period, which costs each ring case
/// about a second and nothing else.

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/testing/roce_probe.h"
#include "cudaq/realtime/testing/server_process.h"
#include "cudaq/realtime/testing/test_utils.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

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

// gpu_roce's --page-size; the caller's slot size must match it, because the
// caller addresses the service's slots by its own stride.
constexpr std::size_t kSlotSize = 256;
// gpu_roce's --payload-size.  Its frames, and so its responses, are
// sizeof(RPCHeader) + this many bytes.
constexpr std::size_t kPayloadSize = 64;

constexpr auto kServerReadyTimeout = std::chrono::seconds(30);
constexpr auto kResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kNoResponseWindow = std::chrono::milliseconds(500);

// Under the ring shape, gpu_roce's TX kernel runs WQE_NUM (64) threads, thread
// j sending slots j, j + 64, ... masked by the slot count: with fewer than 64
// slots, or a count that is not a power of two, several threads send the same
// slot and the caller gets duplicate responses.
constexpr std::size_t kMinSlots = 64;

std::size_t host_page_size() {
  const long page = ::sysconf(_SC_PAGESIZE);
  return page > 0 ? static_cast<std::size_t>(page) : 4096;
}

// gpu_roce's --num-pages, which the caller's ring matches slot for slot (the
// slot index travels in the immediate).  gpu_roce registers its rx ring as one
// memory region, whose size must be a whole number of host pages, so the ring
// is the fewest whole pages holding kMinSlots: 64 slots (4 pages) with 4 KiB
// pages, 256 slots (1 page) with 64 KiB pages.
unsigned ring_slots() {
  const std::size_t per_page =
      std::max<std::size_t>(1, host_page_size() / kSlotSize);
  const std::size_t pages = (kMinSlots + per_page - 1) / per_page;
  return static_cast<unsigned>(per_page * pages);
}

class GpuRoceTwoProcess : public ::testing::TestWithParam<const char *> {
protected:
  void SetUp() override {
    ASSERT_GE(numSlots, kMinSlots);
    ASSERT_EQ(0u, numSlots & (numSlots - 1))
        << numSlots << " slots: gpu_roce masks slot indices, so the count "
        << "must be a power of two";
    ASSERT_EQ(0u, (numSlots * kSlotSize) % host_page_size())
        << "the ring must be a whole number of host pages";

    topology = roce_test_topology();
    const RoceGate gate = roce_test_gate(topology);
    if (gate.action == RoceGate::Fail)
      FAIL() << gate.reason;
    if (gate.action == RoceGate::Skip)
      GTEST_SKIP() << gate.reason;
    int gpus = 0;
    if (cudaGetDeviceCount(&gpus) != cudaSuccess || gpus == 0)
      GTEST_SKIP() << "no CUDA device; the gpu_roce service runs on the GPU";
    const char *caller_dev = topology.caller.device.c_str();
    const char *caller_ip = topology.caller.ip.c_str();
    unified = std::string(GetParam()) == "unified";

    // [1] The caller's QP first: the service needs its number at creation.
    caller = cpu_roce_create_transceiver(
        caller_dev, 1, 0, kSlotSize, kSlotSize, numSlots, "0.0.0.0", 0, 0, 0, 0,
        CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM, 0, 0);
    ASSERT_NE(nullptr, caller);
    cpu_roce_set_local_ip(caller, caller_ip);
    ASSERT_EQ(1, cpu_roce_setup(caller));

    // [2] The service, peered with that QP.
    std::vector<std::string> argv = {
        CUDAQ_REALTIME_TEST_SERVER_PATH, "--transport=gpu_roce",
        "--path=device", std::string("--dispatch=") + GetParam(),
        "--timeout=120",
        // Bridge options from here on, forwarded verbatim by the server.
        "--", "--device=" + topology.service.device,
        "--peer-ip=" + topology.caller.ip,
        "--remote-qp=" + std::to_string(cpu_roce_get_qp_number(caller)),
        "--gpu=0", "--num-pages=" + std::to_string(numSlots),
        "--page-size=" + std::to_string(kSlotSize),
        "--payload-size=" + std::to_string(kPayloadSize)};
    if (unified)
      argv.push_back("--unified");
    ASSERT_TRUE(server.start(argv, kReadyPrefix, kServerReadyTimeout))
        << "server did not become ready; output:\n"
        << server.output();
    ASSERT_EQ("gpu_roce", field("transport")) << server.output();
    ASSERT_EQ("device", field("path")) << server.output();
    EXPECT_EQ(std::string(GetParam()), field("dispatch")) << server.output();
    ASSERT_EQ(std::to_string(numSlots), field("slots"));
    ASSERT_EQ(std::to_string(kSlotSize), field("slot_size"));
    ASSERT_EQ(topology.caller.ip, field("peer_ip")) << server.output();

    // [3] The service's QP and rkey, from READY.
    unsigned long service_qp = 0, service_rkey = 0;
    try {
      service_qp = std::stoul(field("qp"), nullptr, 0);
      service_rkey = std::stoul(field("rkey"), nullptr, 0);
    } catch (const std::exception &) {
      FAIL() << "unparsable qp/rkey in READY; output:\n" << server.output();
    }
    ASSERT_NE(0u, service_qp) << server.output();

    // [4] Connect.  The service's rx ring is registered at IOVA 0, which is
    //     the peer_rx_base_addr the caller was created with.
    ASSERT_EQ(1, cpu_roce_connect(caller, static_cast<unsigned>(service_qp),
                                  topology.service.ip.c_str(),
                                  static_cast<std::uint32_t>(service_rkey)));
    callerMonitor = std::thread([this] { cpu_roce_blocking_monitor(caller); });

    // UC drops a frame that reaches a QP not yet at RTR, without telling
    // either side.  The server connects and launches its kernels only after
    // READY; give that a moment before the first request.
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
    ASSERT_LE(args.size(), kPayloadSize);
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
    txCursor = (txCursor + 1) % numSlots;
  }

  // Wait for the response in the next RX slot and copy its header out.  The
  // slot stays claimed until releaseResponse().
  bool waitResponse(std::uint32_t request_id, RPCResponse &response) {
    const unsigned slot = rxCursor;
    if (!wait_for_flag(rxFlags, slot, kResponseTimeout)) {
      ADD_FAILURE() << "no response in rx slot " << slot << " for request "
                    << request_id << "; server output:\n"
                    << server.output();
      return false;
    }
    std::memcpy(&response, slot_data(rxData, slot, kSlotSize),
                sizeof(response));
    EXPECT_EQ(RPC_MAGIC_RESPONSE, response.magic);
    EXPECT_EQ(request_id, response.request_id);
    return true;
  }

  void releaseResponse() {
    store_flag(rxFlags, rxCursor, 0);
    rxCursor = (rxCursor + 1) % numSlots;
  }

  void expectIncrement(std::uint32_t request_id, std::string_view sent) {
    RPCResponse response{};
    ASSERT_TRUE(waitResponse(request_id, response));
    EXPECT_EQ(0, response.status);
    EXPECT_EQ(static_cast<std::uint32_t>(sent.size()), response.result_len);

    std::string expected(sent.size(), '\0');
    for (std::size_t i = 0; i < sent.size(); ++i)
      expected[i] = static_cast<char>(sent[i] + 1);
    const auto *payload = reinterpret_cast<const char *>(
        slot_data(rxData, rxCursor, kSlotSize) + sizeof(response));
    EXPECT_EQ(expected, std::string(payload, sent.size()));

    releaseResponse();
  }

  void expectNoResponse() {
    EXPECT_FALSE(wait_for_flag(rxFlags, rxCursor, kNoResponseWindow))
        << "expected no response in rx slot " << rxCursor
        << "; server output:\n"
        << server.output();
  }

  // An undispatchable request: dropped under ring, answered with an error
  // under unified (see the file comment).
  void expectErrorResponse(std::uint32_t request_id) {
    if (!unified) {
      expectNoResponse();
      return;
    }
    RPCResponse response{};
    ASSERT_TRUE(waitResponse(request_id, response));
    EXPECT_EQ(-1, response.status);
    EXPECT_EQ(0u, response.result_len);
    releaseResponse();
  }

  // What the service counts after `handled` good requests and one
  // undispatchable one.
  unsigned long long countWithError(unsigned long long handled) const {
    return unified ? handled + 1 : handled;
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
  const unsigned numSlots = ring_slots();
  bool unified = false;
  cpu_roce_transceiver_t caller = nullptr;
  std::thread callerMonitor;
  std::uint64_t txFlags = 0, txData = 0, rxFlags = 0, rxData = 0;
  unsigned txCursor = 0, rxCursor = 0;
};

// The helpers below fail with ASSERT_*, which only leaves the helper. Wrap each
// call in ASSERT_NO_FATAL_FAILURE so a failure ends the test body instead of
// letting a loop run on with a 5 s response timeout per iteration.
TEST_P(GpuRoceTwoProcess, IncrementsStringPayload) {
  const std::string payload = "hello dispatcher";
  ASSERT_NO_FATAL_FAILURE(postIncrement(/*request_id=*/1, payload));
  ASSERT_NO_FATAL_FAILURE(expectIncrement(/*request_id=*/1, payload));
}

TEST_P(GpuRoceTwoProcess, IncrementsBurstAcrossRingWrap) {
  // Three times around both rings, paced one request at a time: a slot whose
  // receive was not re-armed stalls the run.
  const unsigned requests = 3 * numSlots;
  for (unsigned i = 0; i < requests; ++i) {
    const std::string payload = "burst-" + std::to_string(i);
    ASSERT_NO_FATAL_FAILURE(postIncrement(/*request_id=*/i + 1, payload));
    ASSERT_NO_FATAL_FAILURE(expectIncrement(/*request_id=*/i + 1, payload));
  }
}

TEST_P(GpuRoceTwoProcess, RejectsBadMagic) {
  const std::string payload = "before bad magic";
  ASSERT_NO_FATAL_FAILURE(postIncrement(/*request_id=*/1, payload));
  ASSERT_NO_FATAL_FAILURE(expectIncrement(/*request_id=*/1, payload));

  ASSERT_NO_FATAL_FAILURE(
      postIncrement(/*request_id=*/2, "unframed", /*magic=*/0xdeadbeefu));
  ASSERT_NO_FATAL_FAILURE(expectErrorResponse(/*request_id=*/2));

  ASSERT_NO_FATAL_FAILURE(expectProcessedCount(countWithError(1)));
}

TEST_P(GpuRoceTwoProcess, RejectsUnknownFunctionId) {
  const std::string payload = "before unknown id";
  ASSERT_NO_FATAL_FAILURE(postIncrement(/*request_id=*/1, payload));
  ASSERT_NO_FATAL_FAILURE(expectIncrement(/*request_id=*/1, payload));

  ASSERT_NO_FATAL_FAILURE(
      postIncrement(/*request_id=*/2, "unroutable", RPC_MAGIC_REQUEST,
                    /*function_id=*/fnv1a_hash("no_such_function")));
  ASSERT_NO_FATAL_FAILURE(expectErrorResponse(/*request_id=*/2));

  ASSERT_NO_FATAL_FAILURE(expectProcessedCount(countWithError(1)));
}

TEST_P(GpuRoceTwoProcess, ReportsProcessedCount) {
  constexpr unsigned kRequests = 5;
  for (unsigned i = 0; i < kRequests; ++i) {
    const std::string payload = "counted-" + std::to_string(i);
    ASSERT_NO_FATAL_FAILURE(postIncrement(/*request_id=*/i + 1, payload));
    ASSERT_NO_FATAL_FAILURE(expectIncrement(/*request_id=*/i + 1, payload));
  }
  ASSERT_NO_FATAL_FAILURE(expectProcessedCount(kRequests));
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, GpuRoceTwoProcess, ::testing::Values("ring", "unified"),
    [](const ::testing::TestParamInfo<const char *> &info) {
      return std::string(info.param);
    });

} // namespace
