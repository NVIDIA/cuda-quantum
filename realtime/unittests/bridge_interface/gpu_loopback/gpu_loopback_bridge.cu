/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file gpu_loopback_bridge.cu
/// @brief Test-only bridge provider with a GPU data plane and no NIC.
///
/// Serves `get_gpu_dataplane` over rings in mapped pinned host memory, so the
/// whole run-time link -- the dispatcher's embedded kernel, these hooks and a
/// test's handlers -- runs on any GPU.  The test plays the wire: it writes a
/// request into an RX slot and raises that slot's RX flag, and the response
/// appears in the matching TX slot with its TX flag raised.  It reaches the
/// rings through `get_transport_context(RING_BUFFER)`.
///
/// Bridge options:
///   --explicit-garbage   hand over an image that cannot link
///   --no-device-code     hand over no device code at all
///
/// Like any provider, this library links no realtime library: it uses the
/// realtime headers only.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/device_code/device_code_reader.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <new>
#include <string>

namespace {

constexpr std::uint32_t kNumSlots = 8;
constexpr std::uint32_t kSlotSize = 256;

/// What the hooks read, in device memory.
struct loopback_ctx {
  std::uint8_t *rx_data;
  std::uint8_t *tx_data;
  volatile std::uint64_t *rx_flags;
  volatile std::uint64_t *tx_flags;
  std::uint32_t num_slots;
  std::uint32_t slot_size;
};

struct loopback_session {
  const loopback_ctx *ctx;
  volatile int *shutdown_flag;
  std::uint32_t cursor;
};

static_assert(sizeof(loopback_session) <= CUDAQ_DEV_TRANSPORT_SESSION_BYTES &&
                  alignof(loopback_session) <=
                      CUDAQ_DEV_TRANSPORT_SESSION_ALIGN,
              "loopback_session must fit the kernel's session storage");

struct LoopbackBridge {
  bool explicit_garbage = false;
  bool no_device_code = false;
  // Mapped pinned allocations: host views here, device views in `ring`.
  std::uint8_t *rx_data_host = nullptr;
  std::uint8_t *tx_data_host = nullptr;
  std::uint64_t *rx_flags_host = nullptr;
  std::uint64_t *tx_flags_host = nullptr;
  cudaq_ringbuffer_t ring{};
  loopback_ctx *device_ctx = nullptr;
  cudaq_device_image_t device_code{};
};

// Not a fatbin: the device link refuses it.
const unsigned char garbage_bytes[64] = {0xde, 0xad, 0xbe, 0xef};
const cudaq_device_image_t garbage_image = {garbage_bytes,
                                            sizeof(garbage_bytes)};

template <typename T>
bool alloc_mapped(T **host, T **device, std::size_t bytes) {
  void *h = nullptr;
  void *d = nullptr;
  if (cudaHostAlloc(&h, bytes, cudaHostAllocMapped) != cudaSuccess)
    return false;
  std::memset(h, 0, bytes);
  if (cudaHostGetDevicePointer(&d, h, 0) != cudaSuccess) {
    cudaFreeHost(h);
    return false;
  }
  *host = static_cast<T *>(h);
  *device = static_cast<T *>(d);
  return true;
}

void release(LoopbackBridge *b) {
  cudaFreeHost(b->rx_data_host);
  cudaFreeHost(b->tx_data_host);
  cudaFreeHost(b->rx_flags_host);
  cudaFreeHost(b->tx_flags_host);
  cudaFree(b->device_ctx);
  delete b;
}

} // namespace

//==============================================================================
// Device hooks
//==============================================================================

extern "C" __device__ void *cudaq_dev_enter_kernel(void *storage, void *ctx,
                                                   volatile int *shutdown_flag) {
  auto *s = static_cast<loopback_session *>(storage);
  s->ctx = static_cast<const loopback_ctx *>(ctx);
  s->shutdown_flag = shutdown_flag;
  s->cursor = 0;
  return s->ctx ? s : nullptr;
}

extern "C" __device__ cudaq_rx_dev_status_t
cudaq_dev_rx_poll(void *session, void **out_request, void **out_response) {
  auto *s = static_cast<loopback_session *>(session);
  const loopback_ctx *ctx = s->ctx;
  const std::uint32_t slot = s->cursor;
  while (ctx->rx_flags[slot] == 0) {
    if (*s->shutdown_flag)
      return CUDAQ_RX_DEV_SHUTDOWN;
  }
  // The request's bytes were written before its flag.
  __threadfence_system();
  *out_request = ctx->rx_data + static_cast<std::size_t>(slot) * ctx->slot_size;
  *out_response =
      ctx->tx_data + static_cast<std::size_t>(slot) * ctx->slot_size;
  s->cursor = (slot + 1) % ctx->num_slots;
  return CUDAQ_RX_DEV_READY;
}

extern "C" __device__ void cudaq_dev_tx_publish(void *session, void *response) {
  auto *s = static_cast<loopback_session *>(session);
  const loopback_ctx *ctx = s->ctx;
  const auto slot = static_cast<std::uint32_t>(
      (static_cast<std::uint8_t *>(response) - ctx->tx_data) / ctx->slot_size);
  // The response's bytes must be visible before its flag.
  __threadfence_system();
  ctx->tx_flags[slot] = 1;
  ctx->rx_flags[slot] = 0;
}

//==============================================================================
// Bridge interface
//==============================================================================

extern "C" {

static cudaq_status_t loopback_create(cudaq_realtime_bridge_handle_t *handle,
                                      int argc, char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *b = new (std::nothrow) LoopbackBridge();
  if (!b)
    return CUDAQ_ERR_INTERNAL;
  for (int i = 0; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--explicit-garbage")
      b->explicit_garbage = true;
    else if (arg == "--no-device-code")
      b->no_device_code = true;
  }
  const std::size_t data_bytes = std::size_t{kNumSlots} * kSlotSize;
  const std::size_t flag_bytes = kNumSlots * sizeof(std::uint64_t);
  std::uint8_t *rx_data = nullptr;
  std::uint8_t *tx_data = nullptr;
  std::uint64_t *rx_flags = nullptr;
  std::uint64_t *tx_flags = nullptr;
  if (!alloc_mapped(&b->rx_data_host, &rx_data, data_bytes) ||
      !alloc_mapped(&b->tx_data_host, &tx_data, data_bytes) ||
      !alloc_mapped(&b->rx_flags_host, &rx_flags, flag_bytes) ||
      !alloc_mapped(&b->tx_flags_host, &tx_flags, flag_bytes)) {
    release(b);
    return CUDAQ_ERR_CUDA;
  }
  b->ring.rx_data = rx_data;
  b->ring.tx_data = tx_data;
  b->ring.rx_flags = rx_flags;
  b->ring.tx_flags = tx_flags;
  b->ring.rx_stride_sz = kSlotSize;
  b->ring.tx_stride_sz = kSlotSize;
  b->ring.rx_data_host = b->rx_data_host;
  b->ring.tx_data_host = b->tx_data_host;
  b->ring.rx_flags_host = b->rx_flags_host;
  b->ring.tx_flags_host = b->tx_flags_host;

  const loopback_ctx ctx = {rx_data,  tx_data,   rx_flags,
                            tx_flags, kNumSlots, kSlotSize};
  if (cudaMalloc(&b->device_ctx, sizeof(ctx)) != cudaSuccess ||
      cudaMemcpy(b->device_ctx, &ctx, sizeof(ctx), cudaMemcpyHostToDevice) !=
          cudaSuccess) {
    release(b);
    return CUDAQ_ERR_CUDA;
  }
  *handle = b;
  return CUDAQ_OK;
}

static cudaq_status_t loopback_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  release(static_cast<LoopbackBridge *>(handle));
  return CUDAQ_OK;
}

static cudaq_status_t
loopback_get_transport_context(cudaq_realtime_bridge_handle_t handle,
                               cudaq_realtime_transport_context_t context_type,
                               void *out_context) {
  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  if (context_type != RING_BUFFER)
    return CUDAQ_ERR_UNSUPPORTED;
  *static_cast<cudaq_ringbuffer_t *>(out_context) =
      static_cast<LoopbackBridge *>(handle)->ring;
  return CUDAQ_OK;
}

static cudaq_status_t loopback_noop(cudaq_realtime_bridge_handle_t handle) {
  return handle ? CUDAQ_OK : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t
loopback_get_endpoint_info(cudaq_realtime_bridge_handle_t handle, char *buf,
                           size_t buf_len) {
  if (!handle || !buf || buf_len == 0)
    return CUDAQ_ERR_INVALID_ARG;
  const char info[] = "transport=gpu_loopback";
  if (buf_len < sizeof(info))
    return CUDAQ_ERR_INVALID_ARG;
  std::memcpy(buf, info, sizeof(info));
  return CUDAQ_OK;
}

static cudaq_status_t
loopback_get_ring_geometry(cudaq_realtime_bridge_handle_t handle,
                           uint32_t *out_num_slots, uint32_t *out_slot_size) {
  if (!handle || !out_num_slots || !out_slot_size)
    return CUDAQ_ERR_INVALID_ARG;
  *out_num_slots = kNumSlots;
  *out_slot_size = kSlotSize;
  return CUDAQ_OK;
}

static cudaq_status_t
loopback_get_gpu_dataplane(cudaq_realtime_bridge_handle_t handle,
                           cudaq_gpu_dataplane_t *out) {
  if (!handle || !out)
    return CUDAQ_ERR_INVALID_ARG;
  auto *b = static_cast<LoopbackBridge *>(handle);
  out->ctx = b->device_ctx;
  if (b->no_device_code) {
    out->device_code = nullptr;
    out->device_code_count = 0;
    return CUDAQ_OK;
  }
  if (b->explicit_garbage) {
    out->device_code = &garbage_image;
    out->device_code_count = 1;
    return CUDAQ_OK;
  }
  const cudaq_status_t status =
      cudaq_device_code_of_this_binary(&b->device_code);
  if (status != CUDAQ_OK)
    return status;
  out->device_code = &b->device_code;
  out->device_code_count = 1;
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t iface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      loopback_create,
      loopback_destroy,
      loopback_get_transport_context,
      loopback_noop, // connect
      loopback_noop, // launch
      loopback_noop, // disconnect
      /*get_cpu_dataplane=*/nullptr,
      loopback_get_endpoint_info,
      loopback_get_ring_geometry,
      /*set_function_table=*/nullptr,
      loopback_get_gpu_dataplane,
  };
  return &iface;
}

} // extern "C"
