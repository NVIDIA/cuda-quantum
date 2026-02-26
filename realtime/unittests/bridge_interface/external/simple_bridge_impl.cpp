/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file
/// @brief A simple bridge implementation for testing the external bridge
/// interface. This bridge is based on local CUDA devices.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include <cstdint>
#include <cstring>
#include <iostream>

namespace {
#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      std::stringstream ss;                                                    \
      ss << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "            \
         << cudaGetErrorString(err) << std::endl;                              \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  }

struct HostApiBridgeContext {
  static constexpr std::size_t num_slots_ = 2;
  static constexpr std::size_t slot_size_ = 256;
  volatile uint64_t *rx_flags_host_ = nullptr;
  volatile uint64_t *tx_flags_host_ = nullptr;
  volatile uint64_t *rx_flags_ = nullptr;
  volatile uint64_t *tx_flags_ = nullptr;
  std::uint8_t *rx_data_host_ = nullptr;
  std::uint8_t *tx_data_host_ = nullptr;
  std::uint8_t *rx_data_ = nullptr;
  std::uint8_t *tx_data_ = nullptr;

  ~HostApiBridgeContext() {
    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);
  }

  static void free_ring_buffer(volatile uint64_t *host_flags,
                               std::uint8_t *host_data) {
    if (host_flags)
      cudaFreeHost(const_cast<uint64_t *>(host_flags));
    if (host_data)
      cudaFreeHost(host_data);
  }

  static bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                                   volatile uint64_t **host_flags_out,
                                   volatile uint64_t **device_flags_out,
                                   std::uint8_t **host_data_out,
                                   std::uint8_t **device_data_out) {
    void *host_flags_ptr = nullptr;
    cudaError_t err = cudaHostAlloc(
        &host_flags_ptr, num_slots * sizeof(uint64_t), cudaHostAllocMapped);
    if (err != cudaSuccess)
      return false;

    void *device_flags_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0);
    if (err != cudaSuccess) {
      cudaFreeHost(host_flags_ptr);
      return false;
    }

    void *host_data_ptr = nullptr;
    err = cudaHostAlloc(&host_data_ptr, num_slots * slot_size,
                        cudaHostAllocMapped);
    if (err != cudaSuccess) {
      cudaFreeHost(host_flags_ptr);
      return false;
    }

    void *device_data_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0);
    if (err != cudaSuccess) {
      cudaFreeHost(host_flags_ptr);
      cudaFreeHost(host_data_ptr);
      return false;
    }

    memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

    *host_flags_out = static_cast<volatile uint64_t *>(host_flags_ptr);
    *device_flags_out = static_cast<volatile uint64_t *>(device_flags_ptr);
    *host_data_out = static_cast<std::uint8_t *>(host_data_ptr);
    *device_data_out = static_cast<std::uint8_t *>(device_data_ptr);
    return true;
  }
};
} // namespace

extern "C" {
static cudaq_status_t
local_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                    char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;

  HostApiBridgeContext *ctx = new HostApiBridgeContext();
  if (!ctx) {
    std::cerr << "ERROR: Failed to create HostApiBridgeContext" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  // Just allocate the ring buffer here since this simple bridge doesn't have a
  // real "connect" phase.
  if (!ctx->allocate_ring_buffer(ctx->num_slots_, ctx->slot_size_,
                                 &ctx->rx_flags_host_, &ctx->rx_flags_,
                                 &ctx->rx_data_host_, &ctx->rx_data_) ||
      !ctx->allocate_ring_buffer(ctx->num_slots_, ctx->slot_size_,
                                 &ctx->tx_flags_host_, &ctx->tx_flags_,
                                 &ctx->tx_data_host_, &ctx->tx_data_)) {
    std::cerr << "ERROR: Failed to allocate RX or TX ring buffer" << std::endl;
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }

  *handle = reinterpret_cast<cudaq_realtime_bridge_handle_t>(ctx);
  return CUDAQ_OK;
}

static cudaq_status_t
local_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  HostApiBridgeContext *ctx = reinterpret_cast<HostApiBridgeContext *>(handle);

  delete ctx;
  return CUDAQ_OK;
}

static cudaq_status_t
local_bridge_get_ringbuffer(cudaq_realtime_bridge_handle_t handle,
                            cudaq_ringbuffer_t *out_ringbuffer) {

  if (!handle || !out_ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  HostApiBridgeContext *ctx = reinterpret_cast<HostApiBridgeContext *>(handle);
  if (!ctx->rx_flags_ || !ctx->tx_flags_ || !ctx->rx_data_ || !ctx->tx_data_)
    return CUDAQ_ERR_INTERNAL;
  out_ringbuffer->rx_flags = ctx->rx_flags_;
  out_ringbuffer->tx_flags = ctx->tx_flags_;
  out_ringbuffer->rx_data = ctx->rx_data_;
  out_ringbuffer->tx_data = ctx->tx_data_;
  out_ringbuffer->rx_stride_sz = ctx->slot_size_;
  out_ringbuffer->tx_stride_sz = ctx->slot_size_;

  return CUDAQ_OK;
}

static cudaq_status_t
local_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  // Nothing to do
  return CUDAQ_OK;
}

static cudaq_status_t
local_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  // Nothing to do

  return CUDAQ_OK;
}

static cudaq_status_t
local_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  // Nothing to do
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_local_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      local_bridge_create,
      local_bridge_destroy,
      local_bridge_get_ringbuffer,
      local_bridge_connect,
      local_bridge_launch,
      local_bridge_disconnect,
  };
  return &cudaq_local_bridge_interface;
}
}
