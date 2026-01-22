/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file dispatch_kernel.cuh
/// @brief Header-only dispatch kernel for use by external projects.
///
/// This header contains the dispatch kernel implementation that can be
/// included and compiled by external projects (like cudaqx) along with
/// their decoder implementations. This is necessary because CUDA device
/// function pointers can only work within the same compilation unit.
///
/// Usage:
///   1. Include this header in your .cu file that defines the decoder
///   2. Define your decoder as a DeviceRPCFunction
///   3. Register it in the function table
///   4. Call launch_dispatch_kernel_regular/cooperative

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaq::nvqlink {

//==============================================================================
// Dispatch Kernel Implementation (Header-Only)
//==============================================================================

/// @brief Lookup function in table by function_id.
__device__ inline DeviceRPCFunction dispatch_lookup_function(
    std::uint32_t function_id,
    void** function_table,
    std::uint32_t* function_ids,
    std::size_t func_count) {
  
  for (std::size_t i = 0; i < func_count; ++i) {
    if (function_ids[i] == function_id) {
      return reinterpret_cast<DeviceRPCFunction>(function_table[i]);
    }
  }
  return nullptr;
}

/// @brief Templated dispatch kernel with configurable synchronization.
///
/// This kernel implements a persistent polling loop that:
/// 1. Polls the RX ring buffer for incoming requests
/// 2. Parses the RPCHeader to get function_id and argument length
/// 3. Looks up the handler function in the registry
/// 4. Dispatches to the handler
/// 5. Writes the response to the TX ring buffer
///
/// @tparam KernelType Synchronization type (RegularKernel or CooperativeKernel)
/// @tparam DispatchMode How to invoke handlers (DeviceCallMode or GraphLaunchMode)
template <typename KernelType, typename DispatchMode>
__global__ void dispatch_kernel(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    void** function_table,
    std::uint32_t* function_ids,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots) {
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::uint64_t local_packet_count = 0;
  std::size_t current_slot = 0;
  
  // Persistent polling loop
  while (!(*shutdown_flag)) {
    // Thread 0 polls for incoming data
    if (tid == 0) {
      // Check if there's data in the current RX slot
      std::uint64_t rx_value = rx_flags[current_slot];
      
      if (rx_value != 0) {
        // Data available - rx_value is pointer to data buffer
        void* data_buffer = reinterpret_cast<void*>(rx_value);
        
        // Parse RPC header
        RPCHeader* header = static_cast<RPCHeader*>(data_buffer);
        if (header->magic != RPC_MAGIC_REQUEST) {
          // Invalid framing, drop slot
          __threadfence_system();
          rx_flags[current_slot] = 0;
          continue;
        }
        std::uint32_t function_id = header->function_id;
        std::uint32_t arg_len = header->arg_len;
        
        // Get argument data (immediately after header)
        void* arg_buffer = static_cast<void*>(header + 1);
        
        // Lookup function
        DeviceRPCFunction func = dispatch_lookup_function(
            function_id, function_table, function_ids, func_count);
        
        if (func != nullptr) {
          // Call the function
          std::uint32_t result_len = 0;
          std::uint32_t max_result_len = 1024;
          
          int status = func(arg_buffer, arg_len, max_result_len, &result_len);
          
          // Write response header
          RPCResponse* response = static_cast<RPCResponse*>(data_buffer);
          response->magic = RPC_MAGIC_RESPONSE;
          response->status = status;
          response->result_len = result_len;
          
          // Signal TX buffer ready
          __threadfence_system();
          tx_flags[current_slot] = rx_value;
        }
        
        // Clear RX flag (mark slot as processed)
        __threadfence_system();
        rx_flags[current_slot] = 0;
        
        local_packet_count++;
        
        // Move to next slot
        current_slot = (current_slot + 1) % num_slots;
      }
    }
    
    // Synchronize based on kernel type
    KernelType::sync();
    
    // Periodic fence to ensure memory visibility
    if ((local_packet_count & 0xFF) == 0) {
      __threadfence_system();
    }
  }
  
  // Write stats back (only thread 0)
  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(stats), local_packet_count);
  }
}

//==============================================================================
// Inline Kernel Launch Functions
//==============================================================================

/// @brief Launch the dispatch kernel with RegularKernel + DeviceCallMode.
inline void launch_dispatch_kernel_regular_inline(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    void** function_table,
    std::uint32_t* function_ids,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  
  dispatch_kernel<cudaq::realtime::RegularKernel, 
                  cudaq::realtime::DeviceCallMode>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          rx_flags, tx_flags, function_table, function_ids,
          func_count, shutdown_flag, stats, num_slots);
}

/// @brief Launch the dispatch kernel with CooperativeKernel + DeviceCallMode.
inline void launch_dispatch_kernel_cooperative_inline(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    void** function_table,
    std::uint32_t* function_ids,
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
      &function_ids,
      &func_count,
      const_cast<int**>(&shutdown_flag),
      &stats,
      &num_slots
  };
  
  cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          dispatch_kernel<cudaq::realtime::CooperativeKernel,
                          cudaq::realtime::DeviceCallMode>),
      dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, stream);
}

} // namespace cudaq::nvqlink
