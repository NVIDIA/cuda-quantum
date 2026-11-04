/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_dispatcher_set_bridge.cpp
/// @brief cudaq_dispatcher_set_bridge over the udp provider, in one process.
///
/// The dispatcher states a shape and asks the bridge for the matching data
/// plane, so these tests exercise both host shapes end to end (a request over
/// loopback, answered through the wired data plane) and the refusals that
/// turn a mismatched provider into an error at wiring time.  HOST_CALL only,
/// so no GPU is needed.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <gtest/gtest.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef UDP_BRIDGE_PROVIDER_PATH
#error                                                                         \
    "UDP_BRIDGE_PROVIDER_PATH must be defined (path to the built udp provider .so)"
#endif

// Provided by init_rpc_increment_function_table_host.cpp.
extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries);

namespace {

using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

constexpr std::uint32_t kNumSlots = 4;
constexpr std::uint32_t kSlotSize = 256;

cudaq_realtime_bridge_handle_t make_bridge(bool unified) {
  std::vector<std::string> args = {"--port=0",
                                   "--num-slots=" + std::to_string(kNumSlots),
                                   "--slot-size=" + std::to_string(kSlotSize)};
  if (unified)
    args.push_back("--unified");
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

cudaq_dispatcher_config_t make_config(cudaq_dispatch_path_t path,
                                      cudaq_kernel_type_t kernel_type) {
  cudaq_dispatcher_config_t config{};
  config.dispatch_path = path;
  config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  config.kernel_type = kernel_type;
  return config;
}

class SetBridgeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager), CUDAQ_OK);
    setup_rpc_increment_function_table_host(entries);
    table.entries = entries;
    table.count = 1;
  }

  void TearDown() override {
    // Dispatchers first: a running one is still polling its bridge.
    for (auto *dispatcher : dispatchers)
      cudaq_dispatcher_destroy(dispatcher);
    for (auto bridge : bridges)
      if (bridge)
        EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
    cudaq_dispatch_manager_destroy(manager);
  }

  cudaq_realtime_bridge_handle_t bridge(bool unified) {
    bridges.push_back(make_bridge(unified));
    return bridges.back();
  }

  cudaq_dispatcher_t *dispatcher(const cudaq_dispatcher_config_t &config) {
    cudaq_dispatcher_t *d = nullptr;
    EXPECT_EQ(cudaq_dispatcher_create(manager, &config, &d), CUDAQ_OK);
    dispatchers.push_back(d);
    return d;
  }

  void destroy(cudaq_dispatcher_t *d) {
    EXPECT_EQ(cudaq_dispatcher_destroy(d), CUDAQ_OK);
    dispatchers.erase(std::find(dispatchers.begin(), dispatchers.end(), d));
  }

  // Serve one increment request end to end: wire, start, send over
  // loopback, and check the answer and the dispatcher's own count.
  void serveOneRequest(bool unified) {
    auto b = bridge(unified);
    ASSERT_NE(b, nullptr);
    auto *d = dispatcher(
        make_config(CUDAQ_DISPATCH_PATH_HOST,
                    unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR));
    ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_set_function_table(d, &table), CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_set_control(d, &shutdown_flag, &stats),
              CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
    ASSERT_EQ(cudaq_bridge_connect(b), CUDAQ_OK);
    ASSERT_EQ(cudaq_bridge_launch(b), CUDAQ_OK);

    const int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    ASSERT_GE(fd, 0);
    // Bounded, so a missing response fails this test rather than hanging it.
    const timeval reply_timeout{2, 0};
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &reply_timeout,
                 sizeof(reply_timeout));
    sockaddr_in service{};
    service.sin_family = AF_INET;
    service.sin_addr.s_addr = ::htonl(INADDR_LOOPBACK);
    service.sin_port = ::htons(bridgePort(b));

    std::uint8_t request[sizeof(RPCHeader) + 2] = {};
    auto *header = reinterpret_cast<RPCHeader *>(request);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = cudaq::realtime::fnv1a_hash("rpc_increment");
    header->arg_len = 2;
    header->request_id = 7;
    request[sizeof(RPCHeader)] = 41;
    request[sizeof(RPCHeader) + 1] = 99;
    ASSERT_EQ(::sendto(fd, request, sizeof(request), 0,
                       reinterpret_cast<const sockaddr *>(&service),
                       sizeof(service)),
              static_cast<ssize_t>(sizeof(request)));

    std::uint8_t reply[kSlotSize] = {};
    const ssize_t got = ::recv(fd, reply, sizeof(reply), 0);
    ::close(fd);
    ASSERT_GE(got, static_cast<ssize_t>(sizeof(RPCResponse) + 2))
        << "no response from the " << (unified ? "unified" : "ring")
        << " dispatcher";
    // Copied out field by field: the struct is packed, and the assertion
    // macros bind references.
    const auto *response = reinterpret_cast<const RPCResponse *>(reply);
    const std::uint32_t magic = response->magic;
    const std::int32_t status = response->status;
    const std::uint32_t request_id = response->request_id;
    const std::uint32_t result_len = response->result_len;
    EXPECT_EQ(magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
    EXPECT_EQ(status, 0);
    EXPECT_EQ(request_id, 7u);
    ASSERT_EQ(result_len, 2u);
    EXPECT_EQ(reply[sizeof(RPCResponse)], 42);
    EXPECT_EQ(reply[sizeof(RPCResponse) + 1], 100);

    ASSERT_EQ(cudaq_dispatcher_stop(d), CUDAQ_OK);
    std::uint64_t processed = 0;
    ASSERT_EQ(cudaq_dispatcher_get_processed(d, &processed), CUDAQ_OK);
    EXPECT_EQ(processed, 1u);
    EXPECT_EQ(cudaq_bridge_disconnect(b), CUDAQ_OK);
  }

  static std::uint16_t bridgePort(cudaq_realtime_bridge_handle_t b) {
    char endpoint[256] = {};
    EXPECT_EQ(cudaq_bridge_get_endpoint_info(b, endpoint, sizeof(endpoint)),
              CUDAQ_OK);
    // Anchored on the space: "port=" also occurs inside "transport=".
    const char *port = std::strstr(endpoint, " port=");
    EXPECT_NE(port, nullptr) << endpoint;
    return port ? static_cast<std::uint16_t>(std::atoi(port + 6)) : 0;
  }

  cudaq_dispatch_manager_t *manager = nullptr;
  // Members, so they outlive the dispatchers TearDown destroys.
  volatile int shutdown_flag = 0;
  std::uint64_t stats = 0;
  cudaq_function_entry_t entries[1] = {};
  cudaq_function_table_t table{};
  std::vector<cudaq_realtime_bridge_handle_t> bridges;
  std::vector<cudaq_dispatcher_t *> dispatchers;
};

TEST_F(SetBridgeTest, HostRingServesARequest) { serveOneRequest(false); }

TEST_F(SetBridgeTest, HostUnifiedServesARequest) { serveOneRequest(true); }

// The provider serves one shape per configuration; the dispatcher must say so
// when wired, not appear to drop every request later.
TEST_F(SetBridgeTest, RefusesShapesTheProviderDoesNotServe) {
  auto ring_bridge = bridge(false);
  auto unified_bridge = bridge(true);
  ASSERT_NE(ring_bridge, nullptr);
  ASSERT_NE(unified_bridge, nullptr);

  auto *host_unified =
      dispatcher(make_config(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_UNIFIED));
  EXPECT_EQ(cudaq_dispatcher_set_bridge(host_unified, ring_bridge),
            CUDAQ_ERR_UNSUPPORTED);

  auto *host_ring =
      dispatcher(make_config(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_REGULAR));
  EXPECT_EQ(cudaq_dispatcher_set_bridge(host_ring, unified_bridge),
            CUDAQ_ERR_UNSUPPORTED);

  // udp's unified loop runs on the host, so it has nothing for a GPU kernel.
  auto *device_unified =
      dispatcher(make_config(CUDAQ_DISPATCH_PATH_DEVICE, CUDAQ_KERNEL_UNIFIED));
  EXPECT_EQ(cudaq_dispatcher_set_bridge(device_unified, unified_bridge),
            CUDAQ_ERR_UNSUPPORTED);

  // A refusal leaves the dispatchers unwired, so each can still take a bridge
  // that does serve its shape.
  EXPECT_EQ(cudaq_dispatcher_set_bridge(host_unified, unified_bridge),
            CUDAQ_OK);
  EXPECT_EQ(cudaq_dispatcher_set_bridge(host_ring, ring_bridge), CUDAQ_OK);
}

TEST_F(SetBridgeTest, TakesGeometryFromTheBridgeAndRefusesAMismatch) {
  auto b = bridge(false);
  ASSERT_NE(b, nullptr);

  auto mismatched = make_config(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_REGULAR);
  mismatched.num_slots = kNumSlots + 1;
  EXPECT_EQ(cudaq_dispatcher_set_bridge(dispatcher(mismatched), b),
            CUDAQ_ERR_INVALID_ARG);

  // Stating the bridge's own geometry is accepted, the same as leaving it 0.
  auto matching = make_config(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_REGULAR);
  matching.num_slots = kNumSlots;
  matching.slot_size = kSlotSize;
  EXPECT_EQ(cudaq_dispatcher_set_bridge(dispatcher(matching), b), CUDAQ_OK);
}

TEST_F(SetBridgeTest, ADispatcherTakesOneBridge) {
  auto first = bridge(false);
  auto second = bridge(false);
  auto *d =
      dispatcher(make_config(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_REGULAR));
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, first), CUDAQ_OK);
  EXPECT_EQ(cudaq_dispatcher_set_bridge(d, second), CUDAQ_ERR_INVALID_ARG);
}

// In either shape, and whatever shared_ring_mode says.
TEST_F(SetBridgeTest, ABridgeDrivesOneDispatcher) {
  for (const bool unified : {false, true}) {
    SCOPED_TRACE(unified ? "unified" : "ring");
    auto b = bridge(unified);
    auto config =
        make_config(CUDAQ_DISPATCH_PATH_HOST,
                    unified ? CUDAQ_KERNEL_UNIFIED : CUDAQ_KERNEL_REGULAR);
    config.shared_ring_mode = 1;
    auto *first = dispatcher(config);
    auto *second = dispatcher(config);
    ASSERT_EQ(cudaq_dispatcher_set_bridge(first, b), CUDAQ_OK);
    EXPECT_EQ(cudaq_dispatcher_set_bridge(second, b), CUDAQ_ERR_INVALID_ARG);

    // Destroying the first releases the bridge for another dispatcher.
    destroy(first);
    EXPECT_EQ(cudaq_dispatcher_set_bridge(second, b), CUDAQ_OK);
  }
}

} // namespace
