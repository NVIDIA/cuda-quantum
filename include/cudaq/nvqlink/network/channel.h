/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

namespace cudaq::nvqlink {

// Channel execution model determines how packets are delivered to the daemon.
enum class ChannelModel : std::uint8_t {
  POLLING,     // Application polls the channel for packets
  EVENT_DRIVEN // Channel invokes callbacks
};

/// Callback invoked by event-driven channels when packets arrive.
///
/// @param buffers Array of packet buffers
/// @param count Number of buffers
/// @param user_data User context passed during registration
///
using PacketReceivedCallback =
    std::function<void(Buffer **buffers, std::uint32_t count, void *user_data)>;

/// Abstract channel interface for network I/O.
///
/// Channel instances are _NOT_ shareable. That is, multiple threads must not
/// access the same channel instance concurrently.
///
/// Each `Daemon` must create its own `Channel` instance. Multiple channels can
/// share the same physical NIC through the `NetworkDevice` (managed via
/// `NetworkDeviceRegistry`), but each channel owns exclusive access to its
/// allocated queues.
///
/// Example:
/// ```cpp
///   auto channel1 = std::make_unique<ConcreteChannel>(config1);  // Daemon 1
///   auto channel2 = std::make_unique<ConcreteChannel>(config2);  // Daemon 2
///   // channel1 and channel2 share device (via registry), own different queues
/// ```
class Channel {
public:
  Channel(const Channel &) = delete;
  Channel &operator=(const Channel &) = delete;

  virtual ~Channel() = default;

  // Initialization
  virtual void initialize() = 0;
  virtual void cleanup() = 0;

  // Get execution model
  virtual ChannelModel get_execution_model() const = 0;

  // Memory management
  virtual void register_memory(void *addr, std::size_t size) = 0;

  /// Acquire a buffer from the channel's pool (zero allocation on hot path).
  ///
  /// @return Buffer, or nullptr if pool exhausted
  ///
  /// **OWNERSHIP**: Caller acquires ownership of the buffer and MUST either:
  ///   1. Pass to send_burst() (transfers ownership), OR
  ///   2. Call release_buffer() when done
  ///
  /// @note Returns pointer to pre-allocated memory - zero malloc on hot path
  virtual Buffer *acquire_buffer() = 0;

  /// Release a buffer back to the pool.
  ///
  /// @param buffer Buffer to release
  ///
  /// **OWNERSHIP**: Caller must own the buffer. After this call, buffer is
  /// invalid.
  ///
  /// @note Only call if you still own the buffer (didn't pass to send_burst)
  virtual void release_buffer(Buffer *buffer) = 0;

  /// Receive a burst of packets (zero-copy, zero allocation).
  ///
  /// @param buffers Output array for received buffers
  /// @param max_count Maximum buffers to receive
  /// @return Number of buffers received
  ///
  /// **OWNERSHIP**: Channel transfers ownership of buffers to caller.
  ///                Caller MUST call release_buffer() for each buffer when
  ///                done.
  ///
  /// @note Zero-copy: buffers point to pre-allocated ring buffer memory
  virtual std::uint32_t receive_burst(Buffer **buffers,
                                      std::uint32_t max_count) {
    throw std::runtime_error("Polling not supported by this channel");
  }

  /// Send a burst of packets (zero-copy, takes ownership).
  ///
  /// @param buffers Array of buffers to send
  /// @param count Number of buffers to send
  /// @return Number of buffers successfully queued for transmission
  ///
  /// **OWNERSHIP**: Channel takes ownership of ALL buffers (success or
  /// failure).
  ///                Caller MUST NOT touch buffers after this call.
  ///                Channel will release buffers after transmission completes.
  ///
  /// @note Zero-copy: sends directly from pre-allocated memory (no copy)
  virtual std::uint32_t send_burst(Buffer **buffers, std::uint32_t count) {
    throw std::runtime_error("Polling not supported by this channel");
  }

  // Register callback that channel will invoke when packets arrive
  virtual void register_packet_callback(PacketReceivedCallback callback,
                                        void *user_data) {
    throw std::runtime_error("Event-driven mode not supported by this channel");
  }

  // Application must call this to allow channel to process events
  // For DOCA: this calls doca_pe_progress() which invokes callbacks
  virtual void process_events() {
    throw std::runtime_error("Event-driven mode not supported by this channel");
  }

  // GPU mode - returns GPU memory handles for persistent kernel
  struct GPUMemoryHandles {
    void *rx_queue_addr;
    void *tx_queue_addr;
    void *buffer_pool_addr;
    std::size_t buffer_pool_size;
  };
  virtual GPUMemoryHandles get_gpu_memory_handles() {
    return {}; // Only implemented by GPU channels
  }

  // Configuration
  virtual void
  configure_queues(const std::vector<std::uint32_t> &queue_ids) = 0;

protected:
  Channel() = default;
};

} // namespace cudaq::nvqlink
