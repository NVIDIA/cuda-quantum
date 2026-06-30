/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

namespace cudaq::realtime {

/// Optional service-provided synchronizer used when the dispatch loop is owned
/// by the service artifact rather than by a CUDA-Q-created stream.
using DeviceCallDispatchSynchronizeFn = cudaError_t (*)();

/// Dispatch paths supported by a device-call service session.
enum class DeviceCallDispatchMode { Gpu, Host };

/// Dispatch resources exported by one active service session.
struct DeviceCallDispatchTable {
  /// Selects the channel CUDA-Q creates for this session.
  DeviceCallDispatchMode mode = DeviceCallDispatchMode::Gpu;

  /// Function table owned by the session. It must remain valid until `stop()`
  /// completes or the session is destroyed.
  cudaq_function_entry_t *entries = nullptr;

  /// Number of valid entries in `entries`.
  std::uint32_t count = 0;

  /// CUDA device associated with host-dispatch resources. CUDA-Q infers the
  /// device from `entries` for GPU dispatch.
  int deviceId = 0;

  /// Service-owned GPU dispatch-loop launcher. Host dispatch leaves this null.
  cudaq_dispatch_launch_fn_t launchFn = nullptr;

  /// Optional synchronization hook for service-owned asynchronous resources.
  DeviceCallDispatchSynchronizeFn synchronizeFn = nullptr;

  /// Optional pinned, mapped host mailbox used by graph-launch host dispatch.
  /// It is sized for the graph-launch entry count, and the service captures its
  /// device alias when creating each graph executable. CUDA-Q writes the
  /// current graph IO context device pointer into this host mailbox before
  /// launch. A graph-launch handler must write its response through that
  /// context, call `__threadfence_system()`, and publish the context's tx-flag
  /// value as its final operation. This may be null for host-call-only tables.
  void **mailbox = nullptr;
};

/// One active realtime device-call service instance.
///
/// CUDA-Q creates its transport channel from `dispatchTable()`, calls `start()`
/// after the channel is ready, and calls `stop()` after the channel has stopped
/// and released the table resources.
class DeviceCallServiceSession {
public:
  virtual ~DeviceCallServiceSession();

  virtual const DeviceCallDispatchTable &dispatchTable() const noexcept = 0;

  /// Start service-owned processing after channel construction.
  virtual void start() {}

  /// Stop service-owned processing after channel destruction.
  virtual void stop() noexcept {}
};

/// Discoverable provider for active realtime device-call service sessions.
class DeviceCallService {
public:
  virtual ~DeviceCallService();

  /// Create a session for `mode`. A null result means the mode is unsupported.
  virtual std::unique_ptr<DeviceCallServiceSession>
  createDispatchSession(DeviceCallDispatchMode mode) = 0;
};

/// Return a provider that remains valid for every session created from it.
/// CUDA-Q does not take ownership of the returned provider.
using DeviceCallServiceProviderFn = DeviceCallService *(*)();

/// Descriptor returned by a discoverable device-call service plugin.
struct DeviceCallServicePluginInfo {
  const char *pluginName = nullptr;
  DeviceCallServiceProviderFn getService = nullptr;
};

using DeviceCallServicePluginInfoFn = DeviceCallServicePluginInfo (*)();

} // namespace cudaq::realtime

// Default service discovery entry point. Service artifacts may also expose
// suffixed variants with the same signature for tests or multi-service
// deployments, e.g. cudaqGetDeviceCallServicePluginInfo_<name>.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
#endif
extern "C" cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo();
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
