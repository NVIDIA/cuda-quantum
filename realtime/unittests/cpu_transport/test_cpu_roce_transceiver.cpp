/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_cpu_roce_transceiver.cpp
/// @brief Unified (thread-free) mode of the CPU RoCE transceiver C ABI.
///
/// Two halves:
///   - CpuRoceUnifiedContract: the hooks' refusals.  Construction opens no
///     device, so these run on any host.
///   - CpuRoceUnifiedServiceTest: a unified service transceiver driven by hand
///     through cpu_roce_rx_poll / cpu_roce_tx_publish, against an ordinary
///     threaded caller in the same process (the CpuRoceChannel wire pattern:
///     the caller Writes-with-Imm requests, the service Sends responses).  No
///     dispatcher is involved, so these pin the transport's half of the
///     contract on their own.  Needs RoCE: the fixture probes the topology
///     from roce_test_topology() (testing/roce_probe.h) and, when it is not
///     usable, skips -- or fails, when RoCE tests are required (build option
///     CUDAQ_REALTIME_REQUIRE_ROCE_TESTS, env CUDAQ_REALTIME_REQUIRE_ROCE).

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/testing/roce_probe.h"
#include "cudaq/realtime/testing/test_utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <thread>

using cudaq::realtime::testing::load_flag;
using cudaq::realtime::testing::roce_test_gate;
using cudaq::realtime::testing::roce_test_topology;
using cudaq::realtime::testing::RoceGate;
using cudaq::realtime::testing::RoceTopology;
using cudaq::realtime::testing::slot_data;
using cudaq::realtime::testing::store_flag;
using cudaq::realtime::testing::wait_for_flag;

namespace {

constexpr unsigned kNumSlots = 4;
constexpr std::size_t kSlotSize = 256;
constexpr auto kTimeout = std::chrono::milliseconds(5000);

cpu_roce_transceiver_t create_unconnected(int forward, int rx_only, int tx_only,
                                          int unified) {
  return cpu_roce_create_transceiver(
      "not-opened-by-create", /*ib_port=*/1, /*tx_ibv_qp=*/0, kSlotSize,
      kSlotSize, kNumSlots, "0.0.0.0", forward, rx_only, tx_only, unified,
      CPU_ROCE_TX_MODE_RDMA_SEND, 0, 0);
}

TEST(CpuRoceUnifiedContract, UnifiedExcludesTheOtherModes) {
  EXPECT_EQ(nullptr, create_unconnected(1, 0, 0, 1));
  EXPECT_EQ(nullptr, create_unconnected(0, 1, 0, 1));
  EXPECT_EQ(nullptr, create_unconnected(0, 0, 1, 1));
}

TEST(CpuRoceUnifiedContract, HooksRefuseNullHandle) {
  uint32_t slot = 0;
  EXPECT_EQ(0, cpu_roce_rx_poll(nullptr, &slot));
  EXPECT_EQ(0, cpu_roce_tx_publish(nullptr, 0));
}

TEST(CpuRoceUnifiedContract, HooksRefuseBeforeConnect) {
  auto xcvr = create_unconnected(0, 0, 0, /*unified=*/1);
  ASSERT_NE(nullptr, xcvr);
  uint32_t slot = 0;
  EXPECT_EQ(0, cpu_roce_rx_poll(xcvr, &slot));
  EXPECT_EQ(0, cpu_roce_tx_publish(xcvr, 0));
  cpu_roce_destroy_transceiver(xcvr);
}

TEST(CpuRoceUnifiedContract, HooksRefuseTheThreadedShape) {
  auto xcvr = create_unconnected(0, 0, 0, /*unified=*/0);
  ASSERT_NE(nullptr, xcvr);
  uint32_t slot = 0;
  EXPECT_EQ(0, cpu_roce_rx_poll(xcvr, &slot));
  EXPECT_EQ(0, cpu_roce_tx_publish(xcvr, 0));
  cpu_roce_destroy_transceiver(xcvr);
}

class CpuRoceUnifiedServiceTest : public ::testing::Test {
protected:
  void SetUp() override {
    topology = roce_test_topology();
    const RoceGate gate = roce_test_gate(topology);
    if (gate.action == RoceGate::Fail)
      FAIL() << gate.reason;
    if (gate.action == RoceGate::Skip)
      GTEST_SKIP() << gate.reason;
    const std::string &callerIp = topology.caller.ip;
    const std::string &serviceIp = topology.service.ip;

    service = cpu_roce_create_transceiver(
        topology.service.device.c_str(), 1, 0, kSlotSize, kSlotSize, kNumSlots,
        "0.0.0.0", 0, 0, 0, /*unified=*/1, CPU_ROCE_TX_MODE_RDMA_SEND, 0, 0);
    caller = cpu_roce_create_transceiver(
        topology.caller.device.c_str(), 1, 0, kSlotSize, kSlotSize, kNumSlots,
        "0.0.0.0", 0, 0, 0, /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_WRITE_WITH_IMM,
        0, 0);
    ASSERT_NE(nullptr, service);
    ASSERT_NE(nullptr, caller);
    cpu_roce_set_local_ip(service, serviceIp.c_str());
    cpu_roce_set_local_ip(caller, callerIp.c_str());
    ASSERT_EQ(1, cpu_roce_setup(service));
    ASSERT_EQ(1, cpu_roce_setup(caller));

    // In-process stand-in for the TCP rendezvous: the caller Writes into the
    // service's rx ring, so only it needs the peer rkey.
    ASSERT_EQ(1, cpu_roce_connect(service, cpu_roce_get_qp_number(caller),
                                  callerIp.c_str(), 0));
    ASSERT_EQ(1,
              cpu_roce_connect(caller, cpu_roce_get_qp_number(service),
                               serviceIp.c_str(), cpu_roce_get_rkey(service)));
    callerMonitor = std::thread([this] { cpu_roce_blocking_monitor(caller); });

    svcRxFlags = addr(cpu_roce_get_rx_ring_flag_addr(service));
    svcRxData = addr(cpu_roce_get_rx_ring_data_addr(service));
    svcTxFlags = addr(cpu_roce_get_tx_ring_flag_addr(service));
    svcTxData = addr(cpu_roce_get_tx_ring_data_addr(service));
    callerTxFlags = addr(cpu_roce_get_tx_ring_flag_addr(caller));
    callerTxData = addr(cpu_roce_get_tx_ring_data_addr(caller));
    callerRxFlags = addr(cpu_roce_get_rx_ring_flag_addr(caller));
    callerRxData = addr(cpu_roce_get_rx_ring_data_addr(caller));
  }

  void TearDown() override {
    if (caller)
      cpu_roce_close(caller);
    if (callerMonitor.joinable())
      callerMonitor.join();
    if (caller)
      cpu_roce_destroy_transceiver(caller);
    if (service)
      cpu_roce_destroy_transceiver(service);
  }

  template <typename T>
  static std::uint64_t addr(T *p) {
    return reinterpret_cast<std::uint64_t>(p);
  }

  // Caller side: publish `text` from the caller's next TX slot.  The caller's
  // TX pump walks slots in order and writes each into the same-index slot of
  // the service's rx ring.
  void callerSend(const std::string &text) {
    std::uint8_t *tx = slot_data(callerTxData, callerTxCursor, kSlotSize);
    std::memset(tx, 0, kSlotSize);
    std::memcpy(tx, text.data(), text.size());
    store_flag(callerTxFlags, callerTxCursor, addr(tx));
    callerTxCursor = (callerTxCursor + 1) % kNumSlots;
  }

  // Caller side: wait for the next response and return its leading bytes.
  std::string callerReceive(std::size_t len) {
    const unsigned slot = callerRxCursor;
    if (!wait_for_flag(callerRxFlags, slot, kTimeout))
      return "<no response>";
    std::string got(reinterpret_cast<const char *>(
                        slot_data(callerRxData, slot, kSlotSize)),
                    len);
    store_flag(callerRxFlags, slot, 0);
    callerRxCursor = (callerRxCursor + 1) % kNumSlots;
    return got;
  }

  // Service side: spin on the hook until it hands out a slot.
  bool servicePoll(uint32_t &slot,
                   std::chrono::milliseconds timeout = kTimeout) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline)
      if (cpu_roce_rx_poll(service, &slot))
        return true;
    return false;
  }

  // Service side: respond to `slot` with `text` through the publish hook.
  void serviceRespond(uint32_t slot, const std::string &text) {
    std::uint8_t *tx = slot_data(svcTxData, slot, kSlotSize);
    std::memset(tx, 0, kSlotSize);
    std::memcpy(tx, text.data(), text.size());
    ASSERT_EQ(1, cpu_roce_tx_publish(service, slot));
  }

  std::string serviceRequest(uint32_t slot, std::size_t len) const {
    return std::string(
        reinterpret_cast<const char *>(slot_data(svcRxData, slot, kSlotSize)),
        len);
  }

  // A member, not a SetUp() local: the transceivers keep the device-name and
  // peer-IP pointers they are given.
  RoceTopology topology;
  cpu_roce_transceiver_t service = nullptr;
  cpu_roce_transceiver_t caller = nullptr;
  std::thread callerMonitor;
  std::uint64_t svcRxFlags = 0, svcRxData = 0, svcTxFlags = 0, svcTxData = 0;
  std::uint64_t callerTxFlags = 0, callerTxData = 0, callerRxFlags = 0,
                callerRxData = 0;
  unsigned callerTxCursor = 0, callerRxCursor = 0;
};

TEST_F(CpuRoceUnifiedServiceTest, BlockingMonitorStartsNoThreads) {
  // A pump started here would race the hooks for the same CQs; returning at
  // once is the proof that there is none.
  auto done = std::async(std::launch::async,
                         [this] { cpu_roce_blocking_monitor(service); });
  EXPECT_EQ(std::future_status::ready, done.wait_for(std::chrono::seconds(2)));
  if (done.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    cpu_roce_close(service);
}

TEST_F(CpuRoceUnifiedServiceTest, HooksRoundTripToCaller) {
  callerSend("ping");
  uint32_t slot = kNumSlots;
  ASSERT_TRUE(servicePoll(slot));
  EXPECT_EQ(0u, slot);
  EXPECT_EQ("ping", serviceRequest(slot, 4));
  // The unified contract: slot occupancy is reported by return value, never
  // through rx_flags.
  EXPECT_EQ(0u, load_flag(svcRxFlags, slot));

  serviceRespond(slot, "pong");
  EXPECT_EQ("pong", callerReceive(4));
}

TEST_F(CpuRoceUnifiedServiceTest, RxPollFollowsRingOrderAcrossWrap) {
  // Three times around the ring: each slot's recv WQE must be re-armed or the
  // run stalls on the second lap.
  for (unsigned i = 0; i < 3 * kNumSlots; ++i) {
    const std::string request = "req-" + std::to_string(i);
    callerSend(request);
    uint32_t slot = kNumSlots;
    ASSERT_TRUE(servicePoll(slot)) << "request " << i;
    EXPECT_EQ(i % kNumSlots, slot);
    EXPECT_EQ(request, serviceRequest(slot, request.size()));

    const std::string response = "rsp-" + std::to_string(i);
    serviceRespond(slot, response);
    EXPECT_EQ(response, callerReceive(response.size()));
  }
}

TEST_F(CpuRoceUnifiedServiceTest, RxPollHoldsRequestWhileTxFlagSet) {
  // Stand-in for a graph still running on slot 0: its tx_flag is non-zero.
  store_flag(svcTxFlags, 0, 1);
  callerSend("held");
  uint32_t slot = kNumSlots;
  EXPECT_FALSE(servicePoll(slot, std::chrono::milliseconds(200)));

  // Held, not lost: it is delivered as soon as the slot frees.
  store_flag(svcTxFlags, 0, 0);
  ASSERT_TRUE(servicePoll(slot));
  EXPECT_EQ(0u, slot);
  EXPECT_EQ("held", serviceRequest(slot, 4));
}

TEST_F(CpuRoceUnifiedServiceTest, TxPublishRejectsOutOfRangeSlot) {
  EXPECT_EQ(0, cpu_roce_tx_publish(service, kNumSlots));
}

} // namespace
