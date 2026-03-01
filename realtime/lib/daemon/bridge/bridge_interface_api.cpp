/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_interface_api.cpp
/// @brief API implementation for transport layer bridge interface.
///
/// This file provides the implementation of the API functions declared in
/// bridge_interface.h.  It manages the loading of transport provider shared
/// libraries, retrieval of their interface structs, and dispatch of API calls
/// to the appropriate provider implementation based on the bridge handle.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include <dlfcn.h>
#include <iostream>
#include <unordered_map>

namespace {
std::unordered_map<cudaq_realtime_transport_provider_t,
                   cudaq_realtime_bridge_interface_t *>
    provider_interface_map;
std::unordered_map<cudaq_realtime_bridge_handle_t,
                   cudaq_realtime_bridge_interface_t *>
    bridge_handle_interface_map;
/// @brief Path to the built-in Hololink bridge library.  This is used when the
/// provider is CUDAQ_PROVIDER_HOLOLINK to load the Hololink implementation of
/// the bridge interface.  The library must be present at the load path (e.g.,
/// LD_LIBRARY_PATH) for the built-in provider to work.
const char *Hololink_Bridge_Lib = "libcudaq-realtime-bridge-hololink.so";
} // namespace

cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv) {
  const auto it = provider_interface_map.find(provider);
  if (it != provider_interface_map.end()) {
    auto *bridge_interface = it->second;
    return bridge_interface->create(out_bridge_handle, argc, argv);
  }

  const std::string lib_name = [&]() {
    if (provider == CUDAQ_PROVIDER_HOLOLINK) {
      return Hololink_Bridge_Lib;
    } else {
      const char *bridgeLibPath = std::getenv("CUDAQ_REALTIME_BRIDGE_LIB");
      if (!bridgeLibPath) {
        std::cerr << "ERROR: CUDAQ_REALTIME_BRIDGE_LIB environment variable "
                     "not set for EXTERNAL provider"
                  << std::endl;
        return "";
      }
      return bridgeLibPath;
    }
  }();

  if (lib_name.empty())
    return CUDAQ_ERR_INVALID_ARG;
  dlerror(); // reset errors

  if (!out_bridge_handle)
    return CUDAQ_ERR_INVALID_ARG;

  void *lib_handle = dlopen(lib_name.c_str(), RTLD_NOW);

  if (!lib_handle) {
    std::cerr << "ERROR: Failed to load bridge library '" << lib_name
              << "': " << dlerror() << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  using GetInterfaceFunction = cudaq_realtime_bridge_interface_t *(*)();
  GetInterfaceFunction fcn = (GetInterfaceFunction)(intptr_t)dlsym(
      lib_handle, "cudaq_realtime_get_bridge_interface");
  if (!fcn) {
    std::cerr << "ERROR: Failed to interface getter from '" << lib_name
              << "': " << dlerror() << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  cudaq_realtime_bridge_interface_t *bridge_interface = fcn();

  if (!bridge_interface) {
    std::cerr << "ERROR: Bridge interface getter returned null from '"
              << lib_name << "'" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  provider_interface_map[provider] = bridge_interface;

  // Check interface version compatibility
  if (bridge_interface->version != CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION) {
    std::cerr << "ERROR: Bridge interface version mismatch for '" << lib_name
              << "': expected " << CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION
              << ", got " << bridge_interface->version << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  // Run the create callback to allow the bridge to perform any initial setup
  const auto status = bridge_interface->create(out_bridge_handle, argc, argv);
  if (status == CUDAQ_OK) {
    bridge_handle_interface_map[*out_bridge_handle] = bridge_interface;
  }
  return status;
}

cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in destroy" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  const auto status = bridge_interface->destroy(bridge);
  if (status == CUDAQ_OK) {
    bridge_handle_interface_map.erase(it);
  }
  return status;
}

// Retrieve the ringbuffer information for the given bridge.
cudaq_status_t
cudaq_bridge_get_ringbuffer(cudaq_realtime_bridge_handle_t bridge,
                            cudaq_ringbuffer_t *out_ringbuffer) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in get_ringbuffer" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->get_ringbuffer(bridge, out_ringbuffer);
}

cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in connect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->connect(bridge);
}

cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in launch" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->launch(bridge);
}

cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in disconnect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->disconnect(bridge);
}
