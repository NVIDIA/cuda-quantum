/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "cudaq_internal/device_call/DeviceCallTypes.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq_internal::device_call {

// Default ring-buffer slot configuration: two 256-byte slots.
constexpr std::uint32_t DefaultNumSlots = 2;
constexpr std::uint64_t DefaultSlotSize = 256;
// Channel wait timeout, e.g., for waiting for a request slot to be available or
// a response to be ready.
constexpr std::uint64_t DefaultTimeoutMs = 5000;

// Transport-independent view of the realtime dispatcher ring-buffer sizing.
// Concrete channels map these fields to cudaq_dispatcher_config_t and the
// rx/tx slot strides in cudaq_ringbuffer_t.
struct DeviceCallChannelConfig {
  std::uint32_t numSlots = DefaultNumSlots;
  std::uint64_t slotSize = DefaultSlotSize;
  std::uint64_t timeoutMs = DefaultTimeoutMs;
};

// Context for a channel construction
struct DeviceCallChannelCreateArgs {
  // Realtime dispatch table.
  cudaq_function_entry_t *functionTable = nullptr;
  // Number of entries in functionTable.
  std::uint32_t functionCount = 0;
  // CUDA device that owns the dispatch table and backing transport resources.
  int deviceId = 0;
  // Starts the realtime dispatch loop for local shared-memory channels.
  cudaq_dispatch_launch_fn_t launchFn = nullptr;
  // Optional service-provided synchronization hook for dispatch shutdown.
  DeviceCallDispatchSynchronizeFn synchronizeFn = nullptr;
  // Pinned host mailbox used by realtime host-dispatch graph launch. The host
  // dispatcher fills `mailbox[worker_id]` with a GraphIOContext device pointer
  // before launching the corresponding graph handler; the graph kernel reads
  // it via the mailbox's device alias. Required for host_dispatch channels
  // (allocate with `cudaHostAlloc(..., cudaHostAllocMapped)` and size for the
  // GRAPH_LAUNCH worker count). Unused (and left null) by GPU-dispatch
  // channels.
  //
  // Each graph_exec attached to a host-dispatch entry must signal completion
  // with `__threadfence_system(); *io_ctx->tx_flag = io_ctx->tx_flag_value;`
  // because CUDA-Q polls the realtime tx_flag and only returns to the caller
  // once CUDAQ_TX_READY is observed.
  void **mailbox = nullptr;
  // Registry/plugin channel name selected by the runtime configuration.
  std::string channelName;
  // Raw device_call command-line arguments for channel-specific parsing.
  std::vector<std::string> arguments;
  // Transport sizing and wait policy.
  DeviceCallChannelConfig channelConfig;
};

class DeviceCallChannel
    : public cudaq::registry::RegisteredType<DeviceCallChannel> {
public:
  virtual ~DeviceCallChannel() = default;

  // Opaque handle to a transport buffer (e.g. a slot in a ring buffer), which
  // contains a host-accessible data pointer (e.g., pinned memory) and its
  // capacity in bytes.
  struct TransportBuffer {
    std::byte *data = nullptr;
    std::uint64_t capacity = 0;
  };

  // One leased request/response exchange. The channel owns channelPrivate.
  struct DeviceCallFrame {
    std::uint32_t functionId = 0;
    TransportBuffer request;
    TransportBuffer response;
    void *channelPrivate = nullptr;
  };

  // Initialize the channel with the service function table and transport
  // settings.
  virtual void initialize(DeviceCallChannelCreateArgs &&args) = 0;

  // Reserve writable request/response storage for one device_call RPC.
  virtual void acquireFrame(std::uint32_t functionId,
                            std::uint64_t requestBytes,
                            std::uint64_t responseCapacity,
                            DeviceCallFrame &frame) = 0;

  // Publish the request and, when a response is expected, wait for completion.
  virtual std::uint64_t dispatchFrame(DeviceCallFrame &frame) = 0;

  // Return any transport resources associated with a frame lease.
  virtual void releaseFrame(DeviceCallFrame &frame) noexcept = 0;

  // Stop background dispatch resources and make the channel
  // reusable/destructible.
  virtual void stop() noexcept = 0;
};

// Loads external channel plugins before instantiating via the registry.
std::unique_ptr<DeviceCallChannel>
createDeviceCallChannel(const std::string &name,
                        DeviceCallChannelCreateArgs args);

} // namespace cudaq_internal::device_call
