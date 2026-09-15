/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Loopback tests for the UDP ring transceiver (udp_wrapper.h), the plain-UDP
// counterpart of CpuRoceTransceiver. Everything here is plain CPU + loopback
// sockets: no CUDA, no ibverbs, so these tests run anywhere.

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/testing/test_utils.h"
#include <chrono>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <thread>

using cudaq::realtime::testing::load_flag;
using cudaq::realtime::testing::slot_data;
using cudaq::realtime::testing::store_flag;
using cudaq::realtime::testing::wait_for_flag;
using cudaq::realtime::testing::wait_for_flag_clear;

namespace {

constexpr std::size_t kPageSize = 256;
constexpr unsigned kNumPages = 4;

// Every ring in this file has the same stride, so default it; the oversize test
// passes its own to build a datagram the receiver has to reject.
std::uint8_t *slotData(std::uint64_t dataAddr, unsigned slot,
                       std::size_t stride = kPageSize) {
  return slot_data(dataAddr, slot, stride);
}

// A bound (service) and connected (caller) transceiver pair over loopback,
// both pumping.
class UdpTransceiverPairTest : public ::testing::Test {
protected:
  void SetUp() override {
    service = cpu_udp_create_transceiver(kPageSize, kNumPages);
    ASSERT_NE(nullptr, service);
    ASSERT_EQ(1, cpu_udp_bind(service, /*port=*/0));
    ASSERT_NE(0, cpu_udp_get_port(service));
    ASSERT_EQ(1, cpu_udp_start(service));

    caller = cpu_udp_create_transceiver(kPageSize, kNumPages);
    ASSERT_NE(nullptr, caller);
    ASSERT_EQ(1,
              cpu_udp_connect(caller, "127.0.0.1", cpu_udp_get_port(service)));
    ASSERT_EQ(1, cpu_udp_start(caller));

    serviceRxFlags = cpu_udp_get_rx_ring_flag_addr(service);
    serviceRxData = cpu_udp_get_rx_ring_data_addr(service);
    serviceTxFlags = cpu_udp_get_tx_ring_flag_addr(service);
    serviceTxData = cpu_udp_get_tx_ring_data_addr(service);
    callerRxFlags = cpu_udp_get_rx_ring_flag_addr(caller);
    callerRxData = cpu_udp_get_rx_ring_data_addr(caller);
    callerTxFlags = cpu_udp_get_tx_ring_flag_addr(caller);
    callerTxData = cpu_udp_get_tx_ring_data_addr(caller);
  }

  void TearDown() override {
    cpu_udp_destroy_transceiver(caller);
    cpu_udp_destroy_transceiver(service);
  }

  // Publish `payload` into the caller's TX slot; the caller's TX pump ships
  // it to the service and clears the flag.
  void publishFromCaller(unsigned slot, const std::string &payload) {
    ASSERT_LT(payload.size(), kPageSize);
    std::uint8_t *tx = slotData(callerTxData, slot);
    std::memset(tx, 0, kPageSize);
    std::memcpy(tx, payload.data(), payload.size());
    store_flag(callerTxFlags, slot, reinterpret_cast<std::uint64_t>(tx));
  }

  // Consume the service's RX slot: return its payload and recycle the slot
  // (clear the rx flag) so the RX pump's back-pressure releases it.
  std::string consumeAtService(unsigned slot) {
    std::string payload(
        reinterpret_cast<const char *>(slotData(serviceRxData, slot)));
    store_flag(serviceRxFlags, slot, 0);
    return payload;
  }

  cpu_udp_transceiver_t service = nullptr;
  cpu_udp_transceiver_t caller = nullptr;
  std::uint64_t serviceRxFlags = 0, serviceRxData = 0;
  std::uint64_t serviceTxFlags = 0, serviceTxData = 0;
  std::uint64_t callerRxFlags = 0, callerRxData = 0;
  std::uint64_t callerTxFlags = 0, callerTxData = 0;
};

// A UNIFIED (thread-free) service paired with an ordinary pumping caller. This
// test thread drives the service's data plane through the two hooks, standing
// in for the consumer's dispatch loop. The caller is deliberately unchanged:
// only one end opts into this shape, and the wire between them is identical.
class UdpUnifiedServiceTest : public ::testing::Test {
protected:
  void SetUp() override {
    service = cpu_udp_create_transceiver(kPageSize, kNumPages);
    ASSERT_NE(nullptr, service);
    ASSERT_EQ(1, cpu_udp_set_unified(service, 1));
    ASSERT_EQ(1, cpu_udp_bind(service, /*port=*/0));
    ASSERT_NE(0, cpu_udp_get_port(service));
    ASSERT_EQ(1, cpu_udp_start(service)); // starts no threads in this mode

    caller = cpu_udp_create_transceiver(kPageSize, kNumPages);
    ASSERT_NE(nullptr, caller);
    ASSERT_EQ(1,
              cpu_udp_connect(caller, "127.0.0.1", cpu_udp_get_port(service)));
    ASSERT_EQ(1, cpu_udp_start(caller));

    serviceRxFlags = cpu_udp_get_rx_ring_flag_addr(service);
    serviceRxData = cpu_udp_get_rx_ring_data_addr(service);
    serviceTxFlags = cpu_udp_get_tx_ring_flag_addr(service);
    serviceTxData = cpu_udp_get_tx_ring_data_addr(service);
    callerRxFlags = cpu_udp_get_rx_ring_flag_addr(caller);
    callerTxFlags = cpu_udp_get_tx_ring_flag_addr(caller);
    callerRxData = cpu_udp_get_rx_ring_data_addr(caller);
    callerTxData = cpu_udp_get_tx_ring_data_addr(caller);
  }

  void TearDown() override {
    cpu_udp_destroy_transceiver(caller);
    cpu_udp_destroy_transceiver(service);
  }

  void publishFromCaller(unsigned slot, const std::string &payload) {
    ASSERT_LT(payload.size(), kPageSize);
    std::uint8_t *tx = slotData(callerTxData, slot);
    std::memset(tx, 0, kPageSize);
    std::memcpy(tx, payload.data(), payload.size());
    store_flag(callerTxFlags, slot, reinterpret_cast<std::uint64_t>(tx));
  }

  // cpu_udp_rx_poll is non-blocking, so spin the way a dispatch loop does.
  bool pollService(std::uint32_t *slot, std::chrono::milliseconds budget =
                                            std::chrono::milliseconds(2000)) {
    const auto deadline = std::chrono::steady_clock::now() + budget;
    do {
      if (cpu_udp_rx_poll(service, slot))
        return true;
      std::this_thread::sleep_for(std::chrono::microseconds(200));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

  // For the cases where the *absence* of delivery is the property under test.
  bool serviceStaysEmpty(
      std::chrono::milliseconds window = std::chrono::milliseconds(250)) {
    std::uint32_t slot = 0;
    return !pollService(&slot, window);
  }

  std::string serviceSlotPayload(std::uint32_t slot) const {
    return std::string(
        reinterpret_cast<const char *>(slotData(serviceRxData, slot)));
  }

  cpu_udp_transceiver_t service = nullptr;
  cpu_udp_transceiver_t caller = nullptr;
  std::uint64_t serviceRxFlags = 0, serviceRxData = 0;
  std::uint64_t serviceTxFlags = 0, serviceTxData = 0;
  std::uint64_t callerRxFlags = 0, callerRxData = 0;
  std::uint64_t callerTxFlags = 0, callerTxData = 0;
};

TEST(UdpTransceiverLifecycle, RejectsInvalidArguments) {
  EXPECT_EQ(nullptr, cpu_udp_create_transceiver(0, kNumPages));
  EXPECT_EQ(nullptr, cpu_udp_create_transceiver(kPageSize, 0));
  EXPECT_EQ(0, cpu_udp_bind(nullptr, 0));
  EXPECT_EQ(0, cpu_udp_connect(nullptr, "127.0.0.1", 1));
  EXPECT_EQ(0, cpu_udp_start(nullptr));
  EXPECT_EQ(0, cpu_udp_get_port(nullptr));
  cpu_udp_close(nullptr); // must not crash
  cpu_udp_destroy_transceiver(nullptr);
}

TEST(UdpTransceiverLifecycle, BindToConfigurableAddress) {
  // Explicit interface address.
  cpu_udp_transceiver_t xcvr = cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, xcvr);
  EXPECT_EQ(1, cpu_udp_bind_to(xcvr, "127.0.0.1", /*port=*/0));
  EXPECT_NE(0, cpu_udp_get_port(xcvr));
  cpu_udp_destroy_transceiver(xcvr);

  // NULL and "" mean loopback.
  xcvr = cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, xcvr);
  EXPECT_EQ(1, cpu_udp_bind_to(xcvr, "", /*port=*/0));
  cpu_udp_destroy_transceiver(xcvr);

  // A malformed address must fail, not fall back silently.
  xcvr = cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, xcvr);
  EXPECT_EQ(0, cpu_udp_bind_to(xcvr, "not-an-address", /*port=*/0));
  cpu_udp_destroy_transceiver(xcvr);
}

TEST(UdpTransceiverLifecycle, DeliversAcrossAnyInterfaceBind) {
  // Service listening on all interfaces is reachable via loopback.
  cpu_udp_transceiver_t service =
      cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, service);
  ASSERT_EQ(1, cpu_udp_bind_to(service, "0.0.0.0", /*port=*/0));
  ASSERT_EQ(1, cpu_udp_start(service));

  cpu_udp_transceiver_t caller =
      cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, caller);
  ASSERT_EQ(1, cpu_udp_connect(caller, "127.0.0.1", cpu_udp_get_port(service)));
  ASSERT_EQ(1, cpu_udp_start(caller));

  const std::uint64_t txFlags = cpu_udp_get_tx_ring_flag_addr(caller);
  const std::uint64_t txData = cpu_udp_get_tx_ring_data_addr(caller);
  std::uint8_t *tx = slotData(txData, 0);
  std::memset(tx, 0, kPageSize);
  std::memcpy(tx, "any-if", 6);
  store_flag(txFlags, 0, reinterpret_cast<std::uint64_t>(tx));

  const std::uint64_t rxFlags = cpu_udp_get_rx_ring_flag_addr(service);
  const std::uint64_t rxData = cpu_udp_get_rx_ring_data_addr(service);
  ASSERT_TRUE(wait_for_flag(rxFlags, 0));
  EXPECT_EQ(0, std::memcmp(slotData(rxData, 0), "any-if", 6));

  cpu_udp_destroy_transceiver(caller);
  cpu_udp_destroy_transceiver(service);
}

TEST(UdpTransceiverLifecycle, StartRequiresSocketAndCloseIsIdempotent) {
  cpu_udp_transceiver_t xcvr = cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, xcvr);
  // No bind/connect yet: no socket, so the pumps must refuse to start.
  EXPECT_EQ(0, cpu_udp_start(xcvr));

  ASSERT_EQ(1, cpu_udp_bind(xcvr, /*port=*/0));
  EXPECT_NE(0, cpu_udp_get_port(xcvr));
  EXPECT_EQ(1, cpu_udp_start(xcvr));
  EXPECT_EQ(0, cpu_udp_start(xcvr)); // already running

  cpu_udp_close(xcvr);
  cpu_udp_close(xcvr); // idempotent
  cpu_udp_destroy_transceiver(xcvr);
}

TEST_F(UdpTransceiverPairTest, DeliversPublishedSlotToServiceRxRing) {
  publishFromCaller(0, "request-0");

  ASSERT_TRUE(wait_for_flag(serviceRxFlags, 0));
  // The RX flag carries the slot's data address, same contract as RoCE.
  EXPECT_EQ(reinterpret_cast<std::uint64_t>(slotData(serviceRxData, 0)),
            load_flag(serviceRxFlags, 0));
  EXPECT_EQ("request-0", consumeAtService(0));
  // The caller's TX pump recycles the published slot.
  EXPECT_TRUE(wait_for_flag_clear(callerTxFlags, 0));
}

TEST_F(UdpTransceiverPairTest, RoundTripsResponseToCallerRxRing) {
  publishFromCaller(0, "ping");
  ASSERT_TRUE(wait_for_flag(serviceRxFlags, 0));
  EXPECT_EQ("ping", consumeAtService(0));

  // Service answers through its own TX ring; responses go to the source of
  // the most recent inbound datagram (the caller).
  std::uint8_t *tx = slotData(serviceTxData, 0);
  std::memset(tx, 0, kPageSize);
  std::memcpy(tx, "pong", 4);
  store_flag(serviceTxFlags, 0, reinterpret_cast<std::uint64_t>(tx));

  ASSERT_TRUE(wait_for_flag(callerRxFlags, 0));
  EXPECT_EQ(0, std::memcmp(slotData(callerRxData, 0), "pong", 4));
  store_flag(callerRxFlags, 0, 0);
}

TEST_F(UdpTransceiverPairTest, FillsRxSlotsInStrictRingOrder) {
  for (unsigned i = 0; i < 2 * kNumPages; ++i) {
    const unsigned slot = i % kNumPages;
    const std::string payload = "msg-" + std::to_string(i);
    publishFromCaller(slot, payload);
    ASSERT_TRUE(wait_for_flag(serviceRxFlags, slot)) << "message " << i;
    EXPECT_EQ(payload, consumeAtService(slot)) << "message " << i;
  }
}

// Regression test for the TX pump's FIFO contract: published slots ship in
// cursor (publish) order, not ascending index order. A publish burst that
// spans the ring wrap -- here slot kNumPages-1 published before slot 0 --
// must arrive at the peer in publish order; an index scan would ship slot 0
// first and reorder fire-and-forget device_calls on the wire.
TEST_F(UdpTransceiverPairTest, ShipsWrappingPublishBurstInFifoOrder) {
  // Advance the caller's TX cursor to the last slot.
  for (unsigned slot = 0; slot < kNumPages - 1; ++slot) {
    publishFromCaller(slot, "warmup-" + std::to_string(slot));
    ASSERT_TRUE(wait_for_flag(serviceRxFlags, slot));
    consumeAtService(slot);
    ASSERT_TRUE(wait_for_flag_clear(callerTxFlags, slot));
  }

  // Publish the wrapped slot 0 FIRST while the TX cursor still waits on slot
  // kNumPages-1: an index-scan TX pump would ship slot 0 immediately, a FIFO
  // pump must hold it until slot kNumPages-1 is published and shipped.
  publishFromCaller(0, "second");
  publishFromCaller(kNumPages - 1, "first");

  // The service's RX ring assigns slots in arrival order, so the payload
  // arriving first lands in the in-order RX slot kNumPages-1.
  ASSERT_TRUE(wait_for_flag(serviceRxFlags, kNumPages - 1));
  EXPECT_EQ("first", consumeAtService(kNumPages - 1));
  ASSERT_TRUE(wait_for_flag(serviceRxFlags, 0));
  EXPECT_EQ("second", consumeAtService(0));
}

TEST_F(UdpTransceiverPairTest, DropsDatagramsLargerThanOwnStride) {
  // A peer with a larger stride ships full-stride datagrams that exceed this
  // end's page size; the RX pump must drop them (both ends must agree on
  // page_size).
  cpu_udp_transceiver_t bigCaller =
      cpu_udp_create_transceiver(2 * kPageSize, kNumPages);
  ASSERT_NE(nullptr, bigCaller);
  ASSERT_EQ(1,
            cpu_udp_connect(bigCaller, "127.0.0.1", cpu_udp_get_port(service)));
  ASSERT_EQ(1, cpu_udp_start(bigCaller));

  const std::uint64_t txFlags = cpu_udp_get_tx_ring_flag_addr(bigCaller);
  const std::uint64_t txData = cpu_udp_get_tx_ring_data_addr(bigCaller);
  std::uint8_t *tx = slotData(txData, 0, 2 * kPageSize);
  std::memset(tx, 0xAB, 2 * kPageSize);
  store_flag(txFlags, 0, reinterpret_cast<std::uint64_t>(tx));

  // The oversized datagram was shipped (TX flag recycled) ...
  ASSERT_TRUE(wait_for_flag_clear(txFlags, 0));
  // ... but never lands in the service's RX ring.
  EXPECT_FALSE(
      wait_for_flag(serviceRxFlags, 0, std::chrono::milliseconds(250)));

  cpu_udp_destroy_transceiver(bigCaller);
}

//==============================================================================
// Unified (thread-free) mode
//==============================================================================

TEST_F(UdpUnifiedServiceTest, StartSpawnsNoPumpThreads) {
  publishFromCaller(0, "queued");
  // The caller's TX pump ships it, so it really is on the wire ...
  ASSERT_TRUE(wait_for_flag_clear(callerTxFlags, 0));

  // ... but no RX pump exists at the service to move it into a slot. A pump
  // would have published the flag well inside this window.
  EXPECT_FALSE(
      wait_for_flag(serviceRxFlags, 0, std::chrono::milliseconds(250)));

  // The datagram was queued, not lost: the hook still finds it.
  std::uint32_t slot = 42;
  ASSERT_TRUE(pollService(&slot));
  EXPECT_EQ(0u, slot);
  EXPECT_EQ("queued", serviceSlotPayload(slot));
}

TEST_F(UdpUnifiedServiceTest, RxPollReportsSlotByValueAndLeavesRxFlagsAlone) {
  publishFromCaller(0, "request-0");

  std::uint32_t slot = 42;
  ASSERT_TRUE(pollService(&slot));
  EXPECT_EQ(0u, slot);
  // The payload is at the address the consumer derives for itself, since this
  // shape carries no address in a flag.
  EXPECT_EQ("request-0", serviceSlotPayload(slot));
  // Nothing in this shape ever clears an rx flag, so the hook must not set
  // one -- that slot would be occupied forever.
  EXPECT_EQ(0u, load_flag(serviceRxFlags, slot));
}

TEST_F(UdpUnifiedServiceTest, HooksRoundTripToCaller) {
  publishFromCaller(0, "ping");
  std::uint32_t slot = 0;
  ASSERT_TRUE(pollService(&slot));
  EXPECT_EQ("ping", serviceSlotPayload(slot));

  // Answer from the TX half of the same slot, as the dispatcher does. The
  // response goes to the source of the most recent inbound datagram.
  std::uint8_t *tx = slotData(serviceTxData, slot);
  std::memset(tx, 0, kPageSize);
  std::memcpy(tx, "pong", 4);
  ASSERT_EQ(1, cpu_udp_tx_publish(service, slot));

  // The caller is an ordinary pumping transceiver, so its RX ring behaves
  // normally: publishing from the unified end is invisible on the wire.
  ASSERT_TRUE(wait_for_flag(callerRxFlags, 0));
  EXPECT_EQ(0, std::memcmp(slotData(callerRxData, 0), "pong", 4));
  store_flag(callerRxFlags, 0, 0);

  // tx_publish does not touch tx_flags: the consumer owns them here, and
  // there is no TX pump to recycle them.
  EXPECT_EQ(0u, load_flag(serviceTxFlags, slot));
}

TEST_F(UdpUnifiedServiceTest, RxPollBackPressuresOnPendingTxFlag) {
  // tx_flags is the only occupancy marker in this shape: non-zero means a
  // response is still in flight for that slot (in production, a running
  // GRAPH_LAUNCH graph), so its RX half must not be handed out again.
  store_flag(serviceTxFlags, 0, 1);
  publishFromCaller(0, "blocked");
  ASSERT_TRUE(wait_for_flag_clear(callerTxFlags, 0));

  EXPECT_TRUE(serviceStaysEmpty());

  // Back-pressure without consuming: releasing the slot releases the datagram
  // that was waiting behind it.
  store_flag(serviceTxFlags, 0, 0);
  std::uint32_t slot = 42;
  ASSERT_TRUE(pollService(&slot));
  EXPECT_EQ(0u, slot);
  EXPECT_EQ("blocked", serviceSlotPayload(slot));
}

TEST_F(UdpUnifiedServiceTest, RxPollFillsSlotsInStrictRingOrder) {
  // Two laps of the ring: a cursor that failed to wrap or a slot that failed
  // to recycle would stall this.
  for (unsigned i = 0; i < 2 * kNumPages; ++i) {
    const unsigned callerSlot = i % kNumPages;
    const std::string payload = "msg-" + std::to_string(i);
    publishFromCaller(callerSlot, payload);
    ASSERT_TRUE(wait_for_flag_clear(callerTxFlags, callerSlot))
        << "message " << i;

    std::uint32_t slot = 42;
    ASSERT_TRUE(pollService(&slot)) << "message " << i;
    EXPECT_EQ(i % kNumPages, slot) << "message " << i;
    EXPECT_EQ(payload, serviceSlotPayload(slot)) << "message " << i;
  }
}

TEST_F(UdpUnifiedServiceTest, DropsOversizeDatagramInUnifiedMode) {
  // Same policy as the RX pump (both ends must agree on page_size), reached
  // through the hook instead: MSG_TRUNC reports the true length even though
  // only page_size bytes were copied.
  cpu_udp_transceiver_t bigCaller =
      cpu_udp_create_transceiver(2 * kPageSize, kNumPages);
  ASSERT_NE(nullptr, bigCaller);
  ASSERT_EQ(1,
            cpu_udp_connect(bigCaller, "127.0.0.1", cpu_udp_get_port(service)));
  ASSERT_EQ(1, cpu_udp_start(bigCaller));

  const std::uint64_t bigTxFlags = cpu_udp_get_tx_ring_flag_addr(bigCaller);
  const std::uint64_t bigTxData = cpu_udp_get_tx_ring_data_addr(bigCaller);
  std::uint8_t *tx = slotData(bigTxData, 0, 2 * kPageSize);
  std::memset(tx, 0xAB, 2 * kPageSize);
  store_flag(bigTxFlags, 0, reinterpret_cast<std::uint64_t>(tx));

  ASSERT_TRUE(wait_for_flag_clear(bigTxFlags, 0)); // shipped ...
  EXPECT_TRUE(serviceStaysEmpty());                // ... and dropped

  // The drop consumed no slot, so a well-sized datagram still lands in slot 0.
  publishFromCaller(0, "after-drop");
  std::uint32_t slot = 42;
  ASSERT_TRUE(pollService(&slot));
  EXPECT_EQ(0u, slot);
  EXPECT_EQ("after-drop", serviceSlotPayload(slot));

  cpu_udp_destroy_transceiver(bigCaller);
}

TEST_F(UdpUnifiedServiceTest, RejectsMisuse) {
  // The shape cannot be changed under a running transceiver: the pump threads
  // and the hooks would contend for the same socket and slots.
  EXPECT_EQ(0, cpu_udp_set_unified(service, 0));

  // Out of range rather than reading past the end of the ring.
  EXPECT_EQ(0, cpu_udp_tx_publish(service, kNumPages));

  // The hooks are refused on a threaded transceiver, so a consumer that wires
  // the wrong shape gets a failure rather than a race with the pumps. (The
  // caller here is connected and pumping, hence has a peer to answer.)
  std::uint32_t slot = 0;
  EXPECT_EQ(0, cpu_udp_rx_poll(caller, &slot));
  EXPECT_EQ(0, cpu_udp_tx_publish(caller, 0));

  EXPECT_EQ(0, cpu_udp_set_unified(nullptr, 1));
  EXPECT_EQ(0, cpu_udp_rx_poll(nullptr, &slot));
  EXPECT_EQ(0, cpu_udp_rx_poll(service, nullptr));
  EXPECT_EQ(0, cpu_udp_tx_publish(nullptr, 0));
}

TEST(UdpUnifiedLifecycle, PublishNeedsAPeerAndSetUnifiedPrecedesStart) {
  cpu_udp_transceiver_t xcvr = cpu_udp_create_transceiver(kPageSize, kNumPages);
  ASSERT_NE(nullptr, xcvr);
  ASSERT_EQ(1, cpu_udp_set_unified(xcvr, 1));
  // Unified mode does not remove the need for a socket.
  EXPECT_EQ(0, cpu_udp_start(xcvr));
  ASSERT_EQ(1, cpu_udp_bind(xcvr, /*port=*/0));
  ASSERT_EQ(1, cpu_udp_start(xcvr));

  // Nothing has arrived, so no peer has been learned and there is nowhere to
  // send a response.
  EXPECT_EQ(0, cpu_udp_tx_publish(xcvr, 0));
  // And the mode is now frozen.
  EXPECT_EQ(0, cpu_udp_set_unified(xcvr, 1));

  cpu_udp_close(xcvr);
  cpu_udp_destroy_transceiver(xcvr);
}

} // namespace
