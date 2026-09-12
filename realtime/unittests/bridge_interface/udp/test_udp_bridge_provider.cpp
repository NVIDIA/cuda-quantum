/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_udp_bridge_provider.cpp
/// @brief Load-time contract test for the UDP bridge provider.
///
/// A plain (non --pinned-rings) UDP bridge must come up with NO CUDA runtime
/// present: the udp wire is CUDA-free by design and CUDA is only touched when
/// --pinned-rings is passed.  This test links only the bridge loader
/// (cudaq-realtime), NOT the CUDA runtime, and loads the built provider by
/// absolute path.  If the provider .so regresses to a dynamic dependency on
/// libcudart, its dlopen fails here on a host without a CUDA runtime, and
/// cudaq_bridge_create_from_library returns an error instead of CUDAQ_OK.
///
/// The dispatch-shape cases below are contract-and-wiring only: this binary
/// links no transport, so it asserts which shapes the provider offers and that
/// the data plane it hands out is connected to a real socket and the rings it
/// advertised -- but not how that data plane behaves.  The behavior of the
/// hooks belongs to the transceiver that implements them and is covered by
/// test_udp_transceiver's `UdpUnifiedServiceTest`; both shapes under a real
/// dispatcher are covered by test_udp_two_process.  A plain UDP socket is
/// enough to stand in for the far end here, needing no library at all.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef UDP_BRIDGE_PROVIDER_PATH
#error                                                                         \
    "UDP_BRIDGE_PROVIDER_PATH must be defined (path to the built udp provider .so)"
#endif

namespace {

// Create a bridge from the built provider with `args`, which the provider
// parses in create().
cudaq_realtime_bridge_handle_t make_bridge(std::vector<std::string> args) {
  std::vector<char *> argv;
  for (auto &a : args)
    argv.push_back(a.data());
  cudaq_realtime_bridge_handle_t bridge = nullptr;
  EXPECT_EQ(cudaq_bridge_create_from_library(&bridge, UDP_BRIDGE_PROVIDER_PATH,
                                             static_cast<int>(argv.size()),
                                             argv.data()),
            CUDAQ_OK);
  return bridge;
}

} // namespace

TEST(UdpBridgeProvider, LoadsAndCreatesPlainBridgeWithoutCudaRuntime) {
  std::vector<std::string> args = {"--port=0", "--num-slots=8",
                                   "--slot-size=256"};
  std::vector<char *> argv;
  for (auto &a : args)
    argv.push_back(a.data());

  cudaq_realtime_bridge_handle_t bridge = nullptr;
  const cudaq_status_t rc = cudaq_bridge_create_from_library(
      &bridge, UDP_BRIDGE_PROVIDER_PATH, static_cast<int>(argv.size()),
      argv.data());
  ASSERT_EQ(rc, CUDAQ_OK)
      << "Failed to load/create the udp bridge provider '"
      << UDP_BRIDGE_PROVIDER_PATH
      << "'. A dynamic dependency on the CUDA runtime (libcudart) makes the "
         "plain (non-pinned) UDP path unloadable on hosts without a CUDA "
         "runtime installed; link the CUDA runtime statically instead.";
  ASSERT_NE(bridge, nullptr);

  // Geometry and endpoint identity come from the provider's v2 queries.
  uint32_t num_slots = 0;
  uint32_t slot_size = 0;
  EXPECT_EQ(cudaq_bridge_get_ring_geometry(bridge, &num_slots, &slot_size),
            CUDAQ_OK);
  EXPECT_EQ(num_slots, 8u);
  EXPECT_EQ(slot_size, 256u);

  char endpoint[256] = {0};
  EXPECT_EQ(cudaq_bridge_get_endpoint_info(bridge, endpoint, sizeof(endpoint)),
            CUDAQ_OK);
  EXPECT_NE(std::strstr(endpoint, "transport=udp"), nullptr);

  // The v3 set_function_table slot is NULL for this provider (the dispatcher
  // owns dispatch, so the transport never reads the table): a well-formed
  // registration must report the capability as succeeding.
  cudaq_function_entry_t entries[1] = {};
  cudaq_function_table_t table = {entries, 1};
  EXPECT_EQ(cudaq_bridge_set_function_table(bridge, &table), CUDAQ_OK);

  // Argument validation happens ahead of the capability lookup.
  EXPECT_EQ(cudaq_bridge_set_function_table(bridge, nullptr),
            CUDAQ_ERR_INVALID_ARG);
  const cudaq_function_table_t no_entries = {nullptr, 1};
  EXPECT_EQ(cudaq_bridge_set_function_table(bridge, &no_entries),
            CUDAQ_ERR_INVALID_ARG);
  const cudaq_function_table_t empty = {entries, 0};
  EXPECT_EQ(cudaq_bridge_set_function_table(bridge, &empty),
            CUDAQ_ERR_INVALID_ARG);

  EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);

  // An unknown handle (here: the destroyed one) is rejected.
  EXPECT_EQ(cudaq_bridge_set_function_table(bridge, &table),
            CUDAQ_ERR_INVALID_ARG);
}

// The 3-thread ring shape and the single-thread unified shape are mutually
// exclusive, and each is refused in the other's mode.  Refusing is the point:
// silently serving the wrong one produces a transport that appears to drop
// every request.
TEST(UdpBridgeProvider, RefusesCpuDataplaneWithoutUnified) {
  cudaq_realtime_bridge_handle_t bridge =
      make_bridge({"--port=0", "--num-slots=8", "--slot-size=256"});
  ASSERT_NE(bridge, nullptr);

  cudaq_cpu_dataplane_t dataplane = {};
  EXPECT_EQ(cudaq_bridge_get_cpu_dataplane(bridge, &dataplane),
            CUDAQ_ERR_UNSUPPORTED);

  // ... while the ring shape is served, as it always has been.
  cudaq_ringbuffer_t ring = {};
  EXPECT_EQ(cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &ring),
            CUDAQ_OK);

  EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
}

TEST(UdpBridgeProvider, ExposesCpuDataplaneWithUnified) {
  cudaq_realtime_bridge_handle_t bridge = make_bridge(
      {"--unified", "--port=0", "--num-slots=8", "--slot-size=256"});
  ASSERT_NE(bridge, nullptr);

  cudaq_cpu_dataplane_t dataplane = {};
  ASSERT_EQ(cudaq_bridge_get_cpu_dataplane(bridge, &dataplane), CUDAQ_OK);

  // Both hooks are required, and the unified loop derives every slot address
  // from the host-view pointers and the strides, so those must be populated
  // too.  rx_flags is deliberately not asserted on: this shape does not use it.
  EXPECT_NE(dataplane.ctx, nullptr);
  EXPECT_NE(dataplane.rx_poll, nullptr);
  EXPECT_NE(dataplane.tx_publish, nullptr);
  EXPECT_NE(dataplane.ring.rx_data_host, nullptr);
  EXPECT_NE(dataplane.ring.tx_data_host, nullptr);
  EXPECT_NE(dataplane.ring.tx_flags_host, nullptr);
  EXPECT_EQ(dataplane.ring.rx_stride_sz, size_t{256});
  EXPECT_EQ(dataplane.ring.tx_stride_sz, size_t{256});

  // Refusing the other shape is what turns a caller's `--dispatch=` mismatch
  // into a named wiring error rather than a transport that appears to drop
  // every request.
  cudaq_ringbuffer_t ring = {};
  EXPECT_EQ(cudaq_bridge_get_transport_context(bridge, RING_BUFFER, &ring),
            CUDAQ_ERR_UNSUPPORTED);

  EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
}

// A consumer that means to serve unified asks get_transport_context first, the
// same call it would make for the ring shape, and keys off launch_fn to learn
// which flavor of unified it got.  Both configurations are asserted, because
// the whole mechanism rests on a provider refusing the shape it is NOT running.
TEST(UdpBridgeProvider, AnswersUnifiedWithNoDispatcherOverride) {
  cudaq_realtime_bridge_handle_t ring_bridge =
      make_bridge({"--port=0", "--num-slots=8", "--slot-size=256"});
  ASSERT_NE(ring_bridge, nullptr);
  cudaq_unified_dispatch_ctx_t unified_ctx = {};
  EXPECT_EQ(
      cudaq_bridge_get_transport_context(ring_bridge, UNIFIED, &unified_ctx),
      CUDAQ_ERR_UNSUPPORTED);
  EXPECT_EQ(cudaq_bridge_destroy(ring_bridge), CUDAQ_OK);

  cudaq_realtime_bridge_handle_t unified_bridge = make_bridge(
      {"--unified", "--port=0", "--num-slots=8", "--slot-size=256"});
  ASSERT_NE(unified_bridge, nullptr);
  // Deliberately dirty: the provider must write both fields, not merely leave
  // them alone, or a caller reusing a struct reads stale bytes as an override
  // and calls into them.
  cudaq_unified_dispatch_ctx_t dirty;
  dirty.launch_fn = reinterpret_cast<cudaq_unified_launch_fn_t>(0xdeadbeef);
  dirty.transport_ctx = reinterpret_cast<void *>(0xfeedface);
  ASSERT_EQ(cudaq_bridge_get_transport_context(unified_bridge, UNIFIED, &dirty),
            CUDAQ_OK);

  // Null means "unified, but the loop is yours": this transport supplies no
  // dispatcher override, so the consumer goes on to get_cpu_dataplane.  A stray
  // non-null here would send it down the set_unified_launch path instead and
  // the data plane would never be installed.
  EXPECT_EQ(dirty.launch_fn, nullptr);
  EXPECT_EQ(dirty.transport_ctx, nullptr);

  EXPECT_EQ(cudaq_bridge_destroy(unified_bridge), CUDAQ_OK);
}

//==============================================================================
// Is the data plane we hand out actually wired to anything?
//==============================================================================

/// A unified bridge plus a plain UDP socket standing in for the far end.  The
/// tests call the two hooks the way `cudaq_host_unified_loop` does, so nothing
/// here needs a dispatcher.
///
/// The point is the wiring, not the behavior: `dataplane.ctx` must be the
/// transceiver handle (passing the bridge context instead would compile and
/// then corrupt memory), and `dataplane.ring` must describe the same rings
/// those hooks read and write.  Both are the kind of mistake that only a real
/// datagram catches.
class UdpUnifiedWiringTest : public ::testing::Test {
protected:
  static constexpr uint32_t kNumSlots = 4;
  static constexpr uint32_t kSlotSize = 256;

  void SetUp() override {
    bridge = make_bridge({"--unified", "--port=0",
                          "--num-slots=" + std::to_string(kNumSlots),
                          "--slot-size=" + std::to_string(kSlotSize)});
    ASSERT_NE(bridge, nullptr);
    ASSERT_EQ(cudaq_bridge_get_cpu_dataplane(bridge, &dp), CUDAQ_OK);

    // The provider binds loopback and reports the port it landed on, which is
    // the only way to find an ephemeral one.
    char endpoint[256] = {};
    ASSERT_EQ(
        cudaq_bridge_get_endpoint_info(bridge, endpoint, sizeof(endpoint)),
        CUDAQ_OK);
    // Anchored on the space: "port=" also occurs inside "transport=".
    const char *port = std::strstr(endpoint, " port=");
    ASSERT_NE(port, nullptr) << endpoint;
    const int port_number = std::atoi(port + 6);
    ASSERT_GT(port_number, 0) << endpoint;
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    service.sin_port = ::htons(static_cast<uint16_t>(port_number));

    peer_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(peer_fd, 0);
    // Bounded, so a missing response fails this test rather than hanging it.
    const timeval reply_timeout{1, 0};
    ::setsockopt(peer_fd, SOL_SOCKET, SO_RCVTIMEO, &reply_timeout,
                 sizeof(reply_timeout));
  }

  void TearDown() override {
    if (peer_fd >= 0)
      ::close(peer_fd);
    EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
  }

  void sendFromPeer(const std::string &payload) {
    ASSERT_EQ(::sendto(peer_fd, payload.data(), payload.size(), 0,
                       reinterpret_cast<const sockaddr *>(&service),
                       sizeof(service)),
              static_cast<ssize_t>(payload.size()));
  }

  // rx_poll is non-blocking, so spin the way the dispatch loop does.
  bool pollUntilReady(uint32_t *slot, unsigned budget_ms = 2000) {
    for (unsigned i = 0; i < budget_ms * 2; ++i) {
      if (dp.rx_poll(dp.ctx, slot) == CUDAQ_RX_READY)
        return true;
      ::usleep(500);
    }
    return false;
  }

  const char *rxSlot(uint32_t slot) const {
    return reinterpret_cast<const char *>(
        dp.ring.rx_data_host + static_cast<size_t>(slot) * kSlotSize);
  }

  void writeTxSlot(uint32_t slot, const std::string &payload) {
    uint8_t *tx = dp.ring.tx_data_host + static_cast<size_t>(slot) * kSlotSize;
    std::memset(tx, 0, kSlotSize);
    std::memcpy(tx, payload.data(), payload.size());
  }

  cudaq_realtime_bridge_handle_t bridge = nullptr;
  cudaq_cpu_dataplane_t dp = {};
  sockaddr_in service = {};
  int peer_fd = -1;
};

TEST_F(UdpUnifiedWiringTest, HooksRoundTripThroughTheAdvertisedRings) {
  sendFromPeer("ping");

  uint32_t slot = 42;
  ASSERT_TRUE(pollUntilReady(&slot));
  EXPECT_EQ(slot, 0u);
  // Read through the advertised ring base and stride, which is the wiring
  // under test: rx_poll put the datagram where the dispatch loop will look.
  EXPECT_STREQ(rxSlot(slot), "ping");

  // Answer from the TX half of the slot the request arrived in, as the
  // dispatcher does.
  writeTxSlot(slot, "pong");
  ASSERT_EQ(dp.tx_publish(dp.ctx, slot), CUDAQ_OK);

  char reply[kSlotSize] = {};
  const ssize_t got = ::recv(peer_fd, reply, sizeof(reply), 0);
  ASSERT_EQ(got, static_cast<ssize_t>(kSlotSize))
      << "one full stride per frame";
  EXPECT_STREQ(reply, "pong");
}

TEST_F(UdpUnifiedWiringTest, LaunchDoesNotStartAPumpThatRacesTheHooks) {
  // The provider deliberately does not special-case launch() for this shape,
  // on the grounds that the transceiver starts no pump threads once unified.
  // That is the whole safety argument, so it is worth pinning: an arriving
  // datagram must still be sitting on the socket afterwards.
  ASSERT_EQ(cudaq_bridge_launch(bridge), CUDAQ_OK);
  sendFromPeer("still-queued");

  // A pump thread would have consumed it into slot 0 and published the flag.
  ::usleep(250000);
  EXPECT_EQ(dp.ring.rx_flags_host[0], 0u);

  uint32_t slot = 0;
  ASSERT_TRUE(pollUntilReady(&slot));
  EXPECT_STREQ(rxSlot(slot), "still-queued");
}

TEST_F(UdpUnifiedWiringTest, HookFailuresReachTheCallerAsErrors) {
  // The hooks translate the transceiver's 1/0 into status enums, and 0 is all
  // they get -- so every failure arrives as ERR_INTERNAL, without the
  // resolution to say INVALID_ARG. That is acceptable because the unified loop
  // only ever publishes a slot rx_poll just handed it, making these
  // unreachable in production; but the translation must not report success.
  EXPECT_EQ(dp.tx_publish(dp.ctx, kNumSlots), CUDAQ_ERR_INTERNAL);
  // Nothing has arrived, so no peer address has been learned yet.
  EXPECT_EQ(dp.tx_publish(dp.ctx, 0), CUDAQ_ERR_INTERNAL);
}
