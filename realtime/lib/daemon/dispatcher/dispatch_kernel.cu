// Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.

#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel.cuh"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaq::nvqlink {

//==============================================================================
// Dispatch Kernel Implementation (compiled into libcudaq-realtime.so)
//==============================================================================

/// @brief Lookup function entry in table by function_id.
__device__ inline const cudaq_function_entry_t* dispatch_lookup_entry(
    std::uint32_t function_id,
    cudaq_function_entry_t* entries,
    std::size_t entry_count) {
  for (std::size_t i = 0; i < entry_count; ++i) {
    if (entries[i].function_id == function_id) {
      return &entries[i];
    }
  }
  return nullptr;
}

/// @brief Templated dispatch kernel with configurable synchronization.
template <typename KernelType, typename DispatchMode>
__global__ void dispatch_kernel(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::uint64_t local_packet_count = 0;
  std::size_t current_slot = 0;

  while (!(*shutdown_flag)) {
    if (tid == 0) {
      std::uint64_t rx_value = rx_flags[current_slot];
      if (rx_value != 0) {
        void* data_buffer = reinterpret_cast<void*>(rx_value);
        RPCHeader* header = static_cast<RPCHeader*>(data_buffer);
        if (header->magic != RPC_MAGIC_REQUEST) {
          __threadfence_system();
          rx_flags[current_slot] = 0;
          continue;
        }

        std::uint32_t function_id = header->function_id;
        std::uint32_t arg_len = header->arg_len;
        void* arg_buffer = static_cast<void*>(header + 1);

        const cudaq_function_entry_t* entry = dispatch_lookup_entry(
            function_id, function_table, func_count);
        if (entry != nullptr && entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
          DeviceRPCFunction func = 
              reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
          std::uint32_t result_len = 0;
          std::uint32_t max_result_len = 1024;
          int status = func(arg_buffer, arg_len, max_result_len, &result_len);

          RPCResponse* response = static_cast<RPCResponse*>(data_buffer);
          response->magic = RPC_MAGIC_RESPONSE;
          response->status = status;
          response->result_len = result_len;

          __threadfence_system();
          tx_flags[current_slot] = rx_value;
        }

        __threadfence_system();
        rx_flags[current_slot] = 0;
        local_packet_count++;
        current_slot = (current_slot + 1) % num_slots;
      }
    }

    KernelType::sync();

    if ((local_packet_count & 0xFF) == 0) {
      __threadfence_system();
    }
  }

  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(stats), local_packet_count);
  }
}

} // namespace cudaq::nvqlink

extern "C" void cudaq_launch_dispatch_kernel_regular(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  cudaq::nvqlink::dispatch_kernel<cudaq::realtime::RegularKernel,
                                  cudaq::realtime::DeviceCallMode>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          rx_flags, tx_flags, function_table, func_count,
          shutdown_flag, stats, num_slots);
}

extern "C" void cudaq_launch_dispatch_kernel_cooperative(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  void* kernel_args[] = {
      const_cast<std::uint64_t**>(&rx_flags),
      const_cast<std::uint64_t**>(&tx_flags),
      &function_table,
      &func_count,
      const_cast<int**>(&shutdown_flag),
      &stats,
      &num_slots
  };

  cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          cudaq::nvqlink::dispatch_kernel<cudaq::realtime::CooperativeKernel,
                                          cudaq::realtime::DeviceCallMode>),
      dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, stream);
}
