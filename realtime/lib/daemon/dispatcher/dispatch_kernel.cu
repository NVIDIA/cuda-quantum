/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel.cuh"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
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

/// @brief Dispatch kernel for DEVICE_CALL mode only (no graph launch support).
/// This kernel does not contain any device-side graph launch code, avoiding
/// compatibility issues on systems where cudaGraphLaunch is not supported.
///
/// Supports symmetric RX/TX data buffers for Hololink compatibility:
/// - RX data address comes from rx_flags[slot] (set by Hololink RX kernel)
/// - TX response is written to tx_data + slot * tx_stride_sz
/// - tx_flags[slot] is set to the TX slot address
template <typename KernelType>
__global__ void dispatch_kernel_device_call_only(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* tx_data,
    std::size_t tx_stride_sz,
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
        // RX data address comes from rx_flags (set by Hololink RX kernel
        // or host test harness to the address of the RX data slot)
        void* rx_slot = reinterpret_cast<void*>(rx_value);
        RPCHeader* header = static_cast<RPCHeader*>(rx_slot);
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
          // Handler processes in-place: reads args from buffer, writes results back
          int status = func(arg_buffer, arg_len, max_result_len, &result_len);

          // Compute TX slot address from symmetric TX data buffer
          std::uint8_t* tx_slot = tx_data + current_slot * tx_stride_sz;

          // Write RPC response header to TX slot
          RPCResponse* response = reinterpret_cast<RPCResponse*>(tx_slot);
          response->magic = RPC_MAGIC_RESPONSE;
          response->status = status;
          response->result_len = result_len;

          // Copy result data from RX buffer (where handler wrote it) to TX slot
          if (result_len > 0) {
            std::uint8_t* src = static_cast<std::uint8_t*>(arg_buffer);
            std::uint8_t* dst = tx_slot + sizeof(RPCResponse);
            for (std::uint32_t b = 0; b < result_len; ++b) {
              dst[b] = src[b];
            }
          }

          __threadfence_system();
          // Signal TX with the TX slot address (symmetric with Hololink TX kernel)
          tx_flags[current_slot] = reinterpret_cast<std::uint64_t>(tx_slot);
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

/// @brief Dispatch kernel supporting both DEVICE_CALL and GRAPH_LAUNCH modes.
/// This kernel includes device-side graph launch code and requires compute capability >= 9.0.
/// NOTE: Graph launch code is conditionally compiled based on __CUDA_ARCH__.
///
/// Supports symmetric RX/TX data buffers for Hololink compatibility.
template <typename KernelType>
__global__ void dispatch_kernel_with_graph(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* tx_data,
    std::size_t tx_stride_sz,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    void** graph_buffer_ptr,
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
        void* rx_slot = reinterpret_cast<void*>(rx_value);
        RPCHeader* header = static_cast<RPCHeader*>(rx_slot);
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
        
        // Compute TX slot address from symmetric TX data buffer
        std::uint8_t* tx_slot = tx_data + current_slot * tx_stride_sz;

        if (entry != nullptr) {
          if (entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
            DeviceRPCFunction func = 
                reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
            std::uint32_t result_len = 0;
            std::uint32_t max_result_len = 1024;
            int status = func(arg_buffer, arg_len, max_result_len, &result_len);

            // Write RPC response to TX slot
            RPCResponse* response = reinterpret_cast<RPCResponse*>(tx_slot);
            response->magic = RPC_MAGIC_RESPONSE;
            response->status = status;
            response->result_len = result_len;

            // Copy result data from RX buffer to TX slot
            if (result_len > 0) {
              std::uint8_t* src = static_cast<std::uint8_t*>(arg_buffer);
              std::uint8_t* dst = tx_slot + sizeof(RPCResponse);
              for (std::uint32_t b = 0; b < result_len; ++b) {
                dst[b] = src[b];
              }
            }

            __threadfence_system();
            tx_flags[current_slot] = reinterpret_cast<std::uint64_t>(tx_slot);
          }
#if __CUDA_ARCH__ >= 900
          else if (entry->dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH) {
            // Update buffer pointer for graph kernel to read
            if (graph_buffer_ptr != nullptr) {
              *graph_buffer_ptr = rx_slot;
              __threadfence_system();
            }
            
            // Launch pre-created graph
            cudaGraphLaunch(entry->handler.graph_exec, cudaStreamGraphFireAndForget);
            
            __threadfence_system();
            tx_flags[current_slot] = reinterpret_cast<std::uint64_t>(tx_slot);
          }
#endif // __CUDA_ARCH__ >= 900
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

//==============================================================================
// Host Launch Functions
//==============================================================================

// Force eager CUDA module loading for the dispatch kernel.
// Call before launching persistent kernels to avoid lazy-loading deadlocks.
extern "C" cudaError_t cudaq_dispatch_kernel_query_occupancy(
    int* out_blocks, uint32_t threads_per_block) {
  int num_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks,
      cudaq::nvqlink::dispatch_kernel_device_call_only<cudaq::realtime::RegularKernel>,
      threads_per_block, 0);
  if (err != cudaSuccess) return err;
  if (out_blocks) *out_blocks = num_blocks;
  return cudaSuccess;
}

extern "C" void cudaq_launch_dispatch_kernel_regular(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* rx_data,
    std::uint8_t* tx_data,
    std::size_t rx_stride_sz,
    std::size_t tx_stride_sz,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  // Use device-call-only kernel (no graph launch support)
  // Note: rx_data/rx_stride_sz are available in the ringbuffer struct but
  // not passed to the kernel since it reads RX addresses from rx_flags.
  (void)rx_data;
  (void)rx_stride_sz;
  cudaq::nvqlink::dispatch_kernel_device_call_only<cudaq::realtime::RegularKernel>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          rx_flags, tx_flags, tx_data, tx_stride_sz,
          function_table, func_count,
          shutdown_flag, stats, num_slots);
}

extern "C" void cudaq_launch_dispatch_kernel_cooperative(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* rx_data,
    std::uint8_t* tx_data,
    std::size_t rx_stride_sz,
    std::size_t tx_stride_sz,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  (void)rx_data;
  (void)rx_stride_sz;
  void* kernel_args[] = {
      const_cast<std::uint64_t**>(&rx_flags),
      const_cast<std::uint64_t**>(&tx_flags),
      &tx_data,
      &tx_stride_sz,
      &function_table,
      &func_count,
      const_cast<int**>(&shutdown_flag),
      &stats,
      &num_slots
  };

  cudaLaunchCooperativeKernel(
      reinterpret_cast<void*>(
          cudaq::nvqlink::dispatch_kernel_device_call_only<cudaq::realtime::CooperativeKernel>),
      dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, stream);
}

//==============================================================================
// Graph-Based Dispatch (Proper Device-Side Graph Launch Support)
//==============================================================================
//
// To use device-side cudaGraphLaunch(), the dispatch kernel itself must be
// running inside a graph execution context. These functions create a graph
// containing the dispatch kernel, instantiate it with cudaGraphInstantiateFlagDeviceLaunch,
// and provide proper launch/cleanup functions.

// Internal storage for graph-based dispatch context
// Parameters must be stored persistently since the graph may execute after
// the create function returns.
struct cudaq_dispatch_graph_context {
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;
  cudaGraphNode_t kernel_node;
  bool is_valid;
  
  // Persistent storage for kernel parameters (must outlive graph execution)
  volatile std::uint64_t* rx_flags;
  volatile std::uint64_t* tx_flags;
  std::uint8_t* tx_data;
  std::size_t tx_stride_sz;
  cudaq_function_entry_t* function_table;
  std::size_t func_count;
  void** graph_buffer_ptr;
  volatile int* shutdown_flag;
  std::uint64_t* stats;
  std::size_t num_slots;
};

extern "C" cudaError_t cudaq_create_dispatch_graph_regular(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    std::uint8_t* rx_data,
    std::uint8_t* tx_data,
    std::size_t rx_stride_sz,
    std::size_t tx_stride_sz,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    void** graph_buffer_ptr,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream,
    cudaq_dispatch_graph_context** out_context) {
  
  (void)rx_data;
  (void)rx_stride_sz;
  cudaError_t err;
  
  // Allocate context with persistent parameter storage
  cudaq_dispatch_graph_context* ctx = new cudaq_dispatch_graph_context();
  ctx->is_valid = false;
  
  // Store parameters persistently in the context
  ctx->rx_flags = rx_flags;
  ctx->tx_flags = tx_flags;
  ctx->tx_data = tx_data;
  ctx->tx_stride_sz = tx_stride_sz;
  ctx->function_table = function_table;
  ctx->func_count = func_count;
  ctx->graph_buffer_ptr = graph_buffer_ptr;
  ctx->shutdown_flag = shutdown_flag;
  ctx->stats = stats;
  ctx->num_slots = num_slots;
  
  // Create graph
  err = cudaGraphCreate(&ctx->graph, 0);
  if (err != cudaSuccess) {
    delete ctx;
    return err;
  }
  
  // Set up kernel parameters - point to persistent storage in context
  cudaKernelNodeParams kernel_params = {};
  void* kernel_args[] = {
      &ctx->rx_flags,
      &ctx->tx_flags,
      &ctx->tx_data,
      &ctx->tx_stride_sz,
      &ctx->function_table,
      &ctx->func_count,
      &ctx->graph_buffer_ptr,
      &ctx->shutdown_flag,
      &ctx->stats,
      &ctx->num_slots
  };
  
  kernel_params.func = reinterpret_cast<void*>(
      cudaq::nvqlink::dispatch_kernel_with_graph<cudaq::realtime::RegularKernel>);
  kernel_params.gridDim = dim3(num_blocks, 1, 1);
  kernel_params.blockDim = dim3(threads_per_block, 1, 1);
  kernel_params.sharedMemBytes = 0;
  kernel_params.kernelParams = kernel_args;
  kernel_params.extra = nullptr;
  
  // Add kernel node to graph
  err = cudaGraphAddKernelNode(&ctx->kernel_node, ctx->graph, nullptr, 0, &kernel_params);
  if (err != cudaSuccess) {
    cudaGraphDestroy(ctx->graph);
    delete ctx;
    return err;
  }
  
  // Instantiate with device launch flag - THIS IS THE KEY!
  err = cudaGraphInstantiate(&ctx->graph_exec, ctx->graph, 
                              cudaGraphInstantiateFlagDeviceLaunch);
  if (err != cudaSuccess) {
    cudaGraphDestroy(ctx->graph);
    delete ctx;
    return err;
  }
  
  // Upload graph to device (required before device-side launch)
  err = cudaGraphUpload(ctx->graph_exec, stream);
  if (err != cudaSuccess) {
    cudaGraphExecDestroy(ctx->graph_exec);
    cudaGraphDestroy(ctx->graph);
    delete ctx;
    return err;
  }
  
  // Synchronize to ensure upload completes
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    cudaGraphExecDestroy(ctx->graph_exec);
    cudaGraphDestroy(ctx->graph);
    delete ctx;
    return err;
  }
  
  ctx->is_valid = true;
  *out_context = ctx;
  return cudaSuccess;
}

extern "C" cudaError_t cudaq_launch_dispatch_graph(
    cudaq_dispatch_graph_context* context,
    cudaStream_t stream) {
  if (context == nullptr || !context->is_valid) {
    return cudaErrorInvalidValue;
  }
  
  // Launch the graph - now device-side cudaGraphLaunch will work!
  return cudaGraphLaunch(context->graph_exec, stream);
}

extern "C" cudaError_t cudaq_destroy_dispatch_graph(
    cudaq_dispatch_graph_context* context) {
  if (context == nullptr) {
    return cudaErrorInvalidValue;
  }
  
  cudaError_t err = cudaSuccess;
  
  if (context->is_valid) {
    cudaError_t err1 = cudaGraphExecDestroy(context->graph_exec);
    cudaError_t err2 = cudaGraphDestroy(context->graph);
    if (err1 != cudaSuccess) err = err1;
    else if (err2 != cudaSuccess) err = err2;
  }
  
  delete context;
  return err;
}
