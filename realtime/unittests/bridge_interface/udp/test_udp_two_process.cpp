/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_udp_two_process.cpp
/// @brief End-to-end two-process test of the UDP transport: a real dispatcher
///        in a child process serving rpc_increment, driven over loopback UDP.
///
/// This is the first test that exercises the whole service path at once --
/// provider, ring buffer, host dispatch loop, RPC framing, HOST_CALL handler --
/// against a caller in a separate process.  The in-process tests either stop at
/// the provider (test_udp_bridge_provider), at the wire (test_udp_transceiver),
/// or at the dispatch loop (test_host_dispatcher).
///
/// The caller is a second UDP transceiver rather than a raw socket, for two
/// reasons: it ships one full slot stride per datagram (getting that wrong is
/// how a hand-rolled client silently loses every request -- see the "Wire
/// behavior" note in udp_wrapper.h), and it gives the same ring contract the
/// production caller uses.  Malformed frames are still expressible, because the
/// test writes the request bytes into the ring slot itself.
///
/// Both rings are strict FIFO, so the fixture tracks a TX and an RX cursor
/// instead of naming slots: a request occupies the next TX slot, a response
/// arrives in the next RX slot.  The two cursors move independently, which is
/// exactly what a dropped request does -- it consumes a TX slot and produces no
/// response -- and it keeps the test bodies free of slot arithmetic.
///
/// CONSEQUENCE FOR THE NEGATIVE CASES: an undispatchable request must be the
/// LAST one a test sends.  The transceiver's TX pump consumes slots in strict
/// cursor order and parks on the first slot whose flag is clear (udp_wrapper.h
/// assumes every slot gets a response, which is true of the device_call caller
/// but not of the dispatcher), while the dispatcher publishes a TX slot only
/// for requests it actually answers.  A drop therefore leaves a permanent gap
/// that stalls every later response on the wire.  The dispatcher itself is
/// unaffected -- it keeps consuming and counting -- so what these tests can
/// verify is that a refused frame produces no response and is not counted as
/// dispatched, measured against a round trip that is known to work.
///
/// That gap is the ring shape's alone: the unified shape publishes by slot
/// index rather than through a FIFO cursor, so a skipped slot costs it
/// nothing.  The tests are written to the stricter of the two anyway, which is
/// what lets one body serve both.
///
/// Parameterized on the dispatch shape, which crosses the process boundary as
/// command-line tokens -- one for the server, one for the bridge.  Every case
/// runs under both: the client half of the wire
/// is byte-identical, since the shape changes only how the server services its
/// own rings -- 3 threads moving bytes between the wire and the ring buffer, or
/// one dispatcher thread driving the wire itself through the provider's
/// rx_poll/tx_publish hooks.  That is the property worth testing here, and the
/// only way to test it is to assert the two are indistinguishable from outside.

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/testing/server_process.h"
#include "cudaq/realtime/testing/test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#ifndef CUDAQ_REALTIME_TEST_SERVER_PATH
#error "CUDAQ_REALTIME_TEST_SERVER_PATH must be defined (path to the server)"
#endif

using cudaq::realtime::fnv1a_hash;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;
using cudaq::realtime::testing::ServerProcess;
using cudaq::realtime::testing::slot_data;
using cudaq::realtime::testing::store_flag;
using cudaq::realtime::testing::wait_for_flag;

namespace {

constexpr const char *kReadyPrefix = "CUDAQ_REALTIME_SERVER_READY";
constexpr const char *kProcessedPrefix = "CUDAQ_REALTIME_SERVER_PROCESSED";
constexpr std::uint32_t kIncrementId = fnv1a_hash("rpc_increment");

// Small enough to keep the burst test quick, large enough that the burst wraps
// the ring several times.
constexpr unsigned kNumSlots = 4;
constexpr std::size_t kSlotSize = 256;

// A round trip over loopback is sub-millisecond; these are backstops, not
// expected waits.
constexpr auto kResponseTimeout = std::chrono::milliseconds(5000);
constexpr auto kNoResponseWindow = std::chrono::milliseconds(500);

class UdpTwoProcess : public ::testing::TestWithParam<const char *> {
protected:
  void SetUp() override {
    const bool unified = std::string(GetParam()) == "unified";
    std::vector<std::string> argv = {
        CUDAQ_REALTIME_TEST_SERVER_PATH,
        // Server options: how this end is wired, plus a backstop in case a
        // failing test leaves the child unreaped.
        "--transport=udp", std::string("--dispatch=") + GetParam(),
        "--timeout=120",
        // Bridge options from here on, forwarded verbatim by the server.
        "--", "--port=0", "--num-slots=" + std::to_string(kNumSlots),
        "--slot-size=" + std::to_string(kSlotSize)};
    // The shape's second token, this one for the bridge: --dispatch= wires the
    // server, this puts the provider in the matching mode. Deliberately not
    // synthesized by the server -- see the shape note in its header.
    if (unified)
      argv.push_back("--unified");
    ASSERT_TRUE(server.start(argv, kReadyPrefix))
        << "server did not become ready; output:\n"
        << server.output();

    // The shape the server actually brought up. Both tokens agreeing is what
    // gets us here at all: a mismatch fails the server's wiring step.
    EXPECT_EQ(std::string(GetParam()), field("dispatch")) << server.output();
    ASSERT_EQ("udp", field("transport")) << server.output();

    // Geometry comes from the handshake: both ends must agree on the slot
    // stride or every datagram is dropped as oversized.
    slots = static_cast<unsigned>(std::stoul(field("slots")));
    slotSize = static_cast<std::size_t>(std::stoul(field("slot_size")));
    ASSERT_EQ(kNumSlots, slots);
    ASSERT_EQ(kSlotSize, slotSize);
    ASSERT_NE(0, server.port()) << server.output();

    caller = cpu_udp_create_transceiver(slotSize, slots);
    ASSERT_NE(nullptr, caller);
    ASSERT_EQ(1, cpu_udp_connect(caller, "127.0.0.1", server.port()));
    ASSERT_EQ(1, cpu_udp_start(caller));

    txFlags = cpu_udp_get_tx_ring_flag_addr(caller);
    txData = cpu_udp_get_tx_ring_data_addr(caller);
    rxFlags = cpu_udp_get_rx_ring_flag_addr(caller);
    rxData = cpu_udp_get_rx_ring_data_addr(caller);
  }

  void TearDown() override {
    // Caller first: no datagrams in flight while the server tears its rings
    // down.
    if (caller)
      cpu_udp_destroy_transceiver(caller);
    server.stop();
  }

  std::string field(const std::string &key) const {
    const auto it = server.fields().find(key);
    return it == server.fields().end() ? std::string{} : it->second;
  }

  // Publish one request from the next TX slot: a 24-byte RPCHeader followed by
  // `args` verbatim.  The framing defaults to a well-formed rpc_increment
  // call; the negative cases override `magic` or `function_id` to post a frame
  // the dispatcher has to refuse.
  void postIncrement(std::uint32_t request_id, std::string_view args,
                     std::uint32_t magic = RPC_MAGIC_REQUEST,
                     std::uint32_t function_id = kIncrementId) {
    ASSERT_LE(sizeof(RPCHeader) + args.size(), slotSize);
    std::uint8_t *tx = slot_data(txData, txCursor, slotSize);
    std::memset(tx, 0, slotSize);

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
    txCursor = (txCursor + 1) % slots;
  }

  // Wait for the next response, verify it is `sent` with every byte
  // incremented, and recycle the slot so the RX pump's back-pressure releases.
  void expectIncrement(std::uint32_t request_id, std::string_view sent) {
    const unsigned slot = rxCursor;
    ASSERT_TRUE(waitForResponse(slot, kResponseTimeout))
        << "no response in rx slot " << slot << " for request " << request_id
        << "; server output:\n"
        << server.output();

    const std::uint8_t *rx = slot_data(rxData, slot, slotSize);
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
    rxCursor = (rxCursor + 1) % slots;
  }

  // The dispatcher consumes an undispatchable slot without producing a
  // response, so the next RX slot must stay empty.
  void expectNoResponse() {
    EXPECT_FALSE(waitForResponse(rxCursor, kNoResponseWindow))
        << "expected no response in rx slot " << rxCursor
        << "; server output:\n"
        << server.output();
  }

  // Shut the server down and check the count it reports on the way out.  Only
  // successful HOST_CALL invocations are counted, so an exact match is also
  // evidence that nothing else was dispatched.
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

  bool waitForResponse(unsigned slot, std::chrono::milliseconds timeout) const {
    return wait_for_flag(rxFlags, slot, timeout);
  }

  ServerProcess server;
  cpu_udp_transceiver_t caller = nullptr;
  unsigned slots = 0;
  std::size_t slotSize = 0;
  std::uint64_t txFlags = 0, txData = 0, rxFlags = 0, rxData = 0;
  unsigned txCursor = 0, rxCursor = 0;
};

TEST_P(UdpTwoProcess, IncrementsStringPayload) {
  const std::string payload = "hello dispatcher";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);
}

TEST_P(UdpTwoProcess, IncrementsBurstAcrossRingWrap) {
  // Three times around both rings: every slot is reused, so a slot that failed
  // to recycle would stall the run rather than pass.
  const unsigned requests = 3 * slots;
  for (unsigned i = 0; i < requests; ++i) {
    const std::string payload = "burst-" + std::to_string(i);
    postIncrement(/*request_id=*/i + 1, payload);
    expectIncrement(/*request_id=*/i + 1, payload);
  }
}

TEST_P(UdpTwoProcess, DropsBadMagic) {
  // Round trip first, so the refusal below is measured against a path proven
  // to work rather than against a server that might never have come up.
  const std::string payload = "before bad magic";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);

  postIncrement(/*request_id=*/2, "unframed", /*magic=*/0xdeadbeefu);
  expectNoResponse();

  // Bad framing retires the slot without reaching a handler, so only the
  // well-formed request counts.
  expectProcessedCount(1);
}

TEST_P(UdpTwoProcess, DropsUnknownFunctionId) {
  const std::string payload = "before unknown id";
  postIncrement(/*request_id=*/1, payload);
  expectIncrement(/*request_id=*/1, payload);

  // Correct framing, no such entry in the function table.
  postIncrement(/*request_id=*/2, "unroutable", RPC_MAGIC_REQUEST,
                /*function_id=*/fnv1a_hash("no_such_function"));
  expectNoResponse();

  expectProcessedCount(1);
}

TEST_P(UdpTwoProcess, ReportsProcessedCount) {
  constexpr unsigned kRequests = 5;
  for (unsigned i = 0; i < kRequests; ++i) {
    const std::string payload = "counted-" + std::to_string(i);
    postIncrement(/*request_id=*/i + 1, payload);
    expectIncrement(/*request_id=*/i + 1, payload);
  }

  // The dispatcher's counter is only final once its loop has exited, which is
  // why the server reports it during shutdown rather than on request.
  expectProcessedCount(kRequests);
}

INSTANTIATE_TEST_SUITE_P(
    Shapes, UdpTwoProcess, ::testing::Values("ring", "unified"),
    [](const ::testing::TestParamInfo<const char *> &info) {
      return std::string(info.param);
    });

} // namespace
