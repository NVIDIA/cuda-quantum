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

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifndef UDP_BRIDGE_PROVIDER_PATH
#error                                                                         \
    "UDP_BRIDGE_PROVIDER_PATH must be defined (path to the built udp provider .so)"
#endif

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
