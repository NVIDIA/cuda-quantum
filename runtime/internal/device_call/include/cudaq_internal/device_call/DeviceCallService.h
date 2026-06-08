/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq_internal/device_call/DeviceCallTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cudaq_internal::device_call {

// Function table metadata used by host dispatch.
//
// Each entry in `entries` must be a CUDAQ_DISPATCH_GRAPH_LAUNCH entry whose
// `graph_exec` follows the GraphIOContext signaling contract:
//
//   - The kernel reads its request from `io_ctx->rx_slot` (where `io_ctx` is
//     obtained by dereferencing `mailbox[worker_id]` in device code).
//   - The kernel writes its `RPCResponse` (and result payload) to
//     `io_ctx->tx_slot`.
//   - As its final operation, the kernel publishes completion with:
//       __threadfence_system();
//       *io_ctx->tx_flag = io_ctx->tx_flag_value;
//     This is required: CUDA-Q polls `cudaq_host_ringbuffer_poll_tx_flag`
//     and only returns once it observes CUDAQ_TX_READY. A kernel that signals
//     only via `cudaStreamSynchronize` (legacy in-place mode) will cause
//     every device_call to time out.
//
// `mailbox` is the host pointer for a pinned, mapped allocation
// (cudaHostAlloc with cudaHostAllocMapped) sized for `count * sizeof(void*)`.
// The DeviceCallService implementation should capture the device alias for
// that allocation when it creates each graph_exec, but pass the host pointer
// here. CUDA-Q gives this host pointer to the dispatcher; before each graph
// launch, the dispatcher writes the per-launch GraphIOContext device pointer
// into mailbox[worker_id]. The graph observes that value through the captured
// device alias. The mailbox is required for host dispatch.
struct DeviceCallHostDispatchTable {
  cudaq_function_entry_t *entries = nullptr;
  std::uint32_t count = 0;
  std::uint32_t deviceId = 0;
  void **mailbox = nullptr;
};

enum class DeviceCallDispatchMode { Gpu, Host };

struct DeviceCallDispatchTable {
  // Selects which channel type CUDA-Q should create for this session. GPU
  // dispatch uses a device function table; host dispatch uses graph-launch
  // entries.
  DeviceCallDispatchMode mode = DeviceCallDispatchMode::Gpu;

  // Function table exported by the service. The session owns its lifetime and
  // must keep it valid until the session is stopped or destroyed.
  cudaq_function_entry_t *entries = nullptr;

  // Number of valid entries in `entries`. A session with zero entries is not
  // usable by the runtime.
  std::uint32_t count = 0;

  // CUDA device associated with host-dispatch resources. For GPU dispatch,
  // CUDA-Q infers the device from `entries` when constructing the channel.
  int deviceId = 0;

  // Service-owned dispatch loop launcher for GPU dispatch. Host dispatch leaves
  // this null because the channel launches graph entries directly.
  cudaq_dispatch_launch_fn_t launchFn = nullptr;

  // Optional synchronization hook used during channel shutdown when the service
  // owns dispatch-loop streams or other asynchronous resources.
  DeviceCallDispatchSynchronizeFn synchronizeFn = nullptr;

  // Optional pinned host mailbox used by host dispatch to pass the current
  // graph IO context to graph-launch handlers. GPU dispatch leaves this null.
  void **mailbox = nullptr;
};

// Represents one active realtime device-call service instance: i.e., the
// exported realtime function table plus any CUDA streams, graph-launch state,
// mailboxes, or synchronization hooks needed while device_call requests can be
// handled.
class DeviceCallServiceSession {
public:
  virtual ~DeviceCallServiceSession() = default;

  // Return the dispatch resources exported by this active service session.
  // The returned table must remain valid until `stop()` completes or the
  // session is destroyed. CUDA-Q uses this table to construct the transport
  // channel that serves device_call requests.
  virtual const DeviceCallDispatchTable &dispatchTable() const noexcept = 0;

  // Enter the run-loop for the session after CUDA-Q has created the
  // channel from `dispatchTable()`. Implementations should start any
  // service-owned streams, worker state, or synchronization resources needed
  // while requests may be in flight.
  virtual void start() = 0;

  // Stop and release session-owned runtime resources.
  virtual void stop() noexcept = 0;
};

// Interface for realtime device-call service
class DeviceCallService {
public:
  virtual ~DeviceCallService() = default;

  // Create one owned dispatch session. Services may override this method to
  // manage their own resources directly. The default implementation adapts the
  // legacy hook-style methods below.
  virtual std::unique_ptr<DeviceCallServiceSession>
  createDispatchSession(DeviceCallDispatchMode mode);

  // Optional service-state lifecycle around one CUDA-Q dispatch session.
  virtual int create([[maybe_unused]] const void *configPayload,
                     [[maybe_unused]] std::size_t configSize) {
    return 0;
  }
  virtual int destroy() noexcept { return 0; }

  // GPU dispatch function-table setup. CUDA-Q owns the table storage; the
  // service fills it with realtime entries for its exported device functions.
  virtual std::uint32_t getFunctionCount() const { return 0; }
  virtual int populateTable([[maybe_unused]] cudaq_function_entry_t *entries,
                            [[maybe_unused]] std::uint32_t capacity,
                            [[maybe_unused]] cudaStream_t stream) {
    return 1;
  }

  // GPU dispatch launch hooks. Services that only support host dispatch may
  // leave these null.
  virtual cudaq_dispatch_launch_fn_t getDeviceDispatchLaunch() const {
    return nullptr;
  }
  virtual DeviceCallDispatchSynchronizeFn getDeviceDispatchSynchronize() const {
    return nullptr;
  }

  // Host-dispatch graph table setup. Services that only support GPU dispatch
  // may return failure.
  virtual int
  getHostDispatchTable([[maybe_unused]] DeviceCallHostDispatchTable &table) {
    return 1;
  }

  // Optional session hooks called around an active CUDA-Q device_call endpoint.
  virtual int start() { return 0; }
  virtual int stop() noexcept { return 0; }
};

using DeviceCallServiceProviderFn = DeviceCallService *(*)();

struct DeviceCallServicePluginInfo {
  const char *pluginName = nullptr;
  DeviceCallServiceProviderFn getService = nullptr;
};

using DeviceCallServicePluginInfoFn = DeviceCallServicePluginInfo (*)();

} // namespace cudaq_internal::device_call

// Default service discovery entry point. Service artifacts may also expose
// suffixed variants with the same signature for tests or multi-service
// deployments, e.g. cudaqGetDeviceCallServicePluginInfo_<name>.
extern "C" cudaq_internal::device_call::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo();
