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
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
namespace {
std::unordered_map<cudaq_realtime_transport_provider_t,
                   cudaq_realtime_bridge_interface_t *>
    provider_interface_map;

std::unordered_map<cudaq_realtime_bridge_handle_t,
                   cudaq_realtime_bridge_interface_t *>
    bridge_handle_interface_map;

// Mutex to protect access to global maps (provider_interface_map and
// bridge_handle_interface_map) for thread safety.
std::shared_mutex bridge_interface_mutex;

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
  // For create, hold an unique lock.
  std::unique_lock<std::shared_mutex> lock(bridge_interface_mutex);

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

  // Check interface version compatibility BEFORE caching the interface:
  // versions older than CURRENT are accepted (fields beyond `disconnect` are
  // guarded by version checks at the call sites); newer versions are rejected
  // since the provider's struct may reference callbacks this library does not
  // know how to drive.
  if (bridge_interface->version < 1 ||
      bridge_interface->version > CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION) {
    std::cerr << "ERROR: Bridge interface version mismatch for '" << lib_name
              << "': this library supports versions 1.."
              << CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION << ", got "
              << bridge_interface->version << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  provider_interface_map[provider] = bridge_interface;
  // Run the create callback to allow the bridge to perform any initial setup
  const auto status = bridge_interface->create(out_bridge_handle, argc, argv);
  if (status == CUDAQ_OK) {
    bridge_handle_interface_map[*out_bridge_handle] = bridge_interface;
  }
  return status;
}

cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge) {
  // For destroy, hold an unique lock.
  std::unique_lock<std::shared_mutex> lock(bridge_interface_mutex);

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

// Retrieve the transport context information for the given bridge.
cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in get_transport_context"
              << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->get_transport_context(bridge, context_type,
                                                 out_context);
}

cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in connect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->connect(bridge);
}

cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in launch" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->launch(bridge);
}

cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge) {
  // Hold a shared lock since this is a read-only operation on the global maps.
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);

  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in disconnect" << std::endl;
    return CUDAQ_ERR_INVALID_ARG;
  }
  auto *bridge_interface = it->second;
  return bridge_interface->disconnect(bridge);
}

//==============================================================================
// Version-2 capability queries.  Fields beyond `disconnect` may only be read
// from providers reporting version >= 2 (a v1 provider's struct may simply
// end at `disconnect`); missing capability => CUDAQ_ERR_UNSUPPORTED.
//==============================================================================

namespace {
// Look up the interface for `bridge` and return it only if it is a v2+
// provider; sets `*status` and returns nullptr otherwise.
cudaq_realtime_bridge_interface_t *
find_v2_interface(cudaq_realtime_bridge_handle_t bridge, const char *what,
                  cudaq_status_t *status) {
  const auto it = bridge_handle_interface_map.find(bridge);
  if (it == bridge_handle_interface_map.end()) {
    std::cerr << "ERROR: Invalid bridge handle in " << what << std::endl;
    *status = CUDAQ_ERR_INVALID_ARG;
    return nullptr;
  }
  if (it->second->version < 2) {
    *status = CUDAQ_ERR_UNSUPPORTED;
    return nullptr;
  }
  *status = CUDAQ_OK;
  return it->second;
}
} // namespace

cudaq_status_t
cudaq_bridge_get_cpu_dataplane(cudaq_realtime_bridge_handle_t bridge,
                               cudaq_cpu_dataplane_t *out_dataplane) {
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);
  cudaq_status_t status;
  auto *bridge_interface =
      find_v2_interface(bridge, "get_cpu_dataplane", &status);
  if (!bridge_interface)
    return status;
  if (!bridge_interface->get_cpu_dataplane)
    return CUDAQ_ERR_UNSUPPORTED;
  return bridge_interface->get_cpu_dataplane(bridge, out_dataplane);
}

cudaq_status_t
cudaq_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t bridge, char *buf,
                               size_t buf_len) {
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);
  cudaq_status_t status;
  auto *bridge_interface =
      find_v2_interface(bridge, "get_endpoint_info", &status);
  if (!bridge_interface)
    return status;
  if (!bridge_interface->get_endpoint_info)
    return CUDAQ_ERR_UNSUPPORTED;
  return bridge_interface->get_endpoint_info(bridge, buf, buf_len);
}

cudaq_status_t
cudaq_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t bridge,
                               uint32_t *out_num_slots,
                               uint32_t *out_slot_size) {
  std::shared_lock<std::shared_mutex> lock(bridge_interface_mutex);
  cudaq_status_t status;
  auto *bridge_interface =
      find_v2_interface(bridge, "get_ring_geometry", &status);
  if (!bridge_interface)
    return status;
  if (!bridge_interface->get_ring_geometry)
    return CUDAQ_ERR_UNSUPPORTED;
  return bridge_interface->get_ring_geometry(bridge, out_num_slots,
                                             out_slot_size);
}
