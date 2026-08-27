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

} // namespace
