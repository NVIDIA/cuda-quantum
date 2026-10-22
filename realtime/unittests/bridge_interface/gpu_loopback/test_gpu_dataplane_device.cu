/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_gpu_dataplane_device.cu
/// @brief test_gpu_dataplane's device code: the handler the run-time linked
/// dispatcher calls, and a kernel that takes the handler's address in this
/// binary's own module, which the dispatcher must refuse.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/device_code/device_code.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <cstddef>
#include <cstdint>

namespace {

// Adds 1 to each payload byte.
__device__ int gpu_dataplane_test_handler(const void *rx_slot, void *tx_slot,
                                          std::size_t slot_size) {
  const auto *header = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  const auto *in = reinterpret_cast<const std::uint8_t *>(header + 1);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  auto *out = reinterpret_cast<std::uint8_t *>(response + 1);
  const std::size_t capacity = slot_size - sizeof(cudaq::realtime::RPCResponse);
  const std::uint32_t len =
      header->arg_len < capacity ? header->arg_len
                                 : static_cast<std::uint32_t>(capacity);
  for (std::uint32_t i = 0; i < len; ++i)
    out[i] = static_cast<std::uint8_t>(in[i] + 1);
  response->result_len = len;
  return 0;
}

__global__ void own_module_address_kernel(void **out) {
  *out = reinterpret_cast<void *>(&gpu_dataplane_test_handler);
}

} // namespace

CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(gpu_dataplane_test_handler)

/// The handler's address in this binary's own module, not the linked one.
extern "C" void *gpu_dataplane_test_own_module_handler() {
  void **d_out = nullptr;
  void *out = nullptr;
  if (cudaMalloc(&d_out, sizeof(void *)) != cudaSuccess)
    return nullptr;
  own_module_address_kernel<<<1, 1>>>(d_out);
  cudaMemcpy(&out, d_out, sizeof(out), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return out;
}
