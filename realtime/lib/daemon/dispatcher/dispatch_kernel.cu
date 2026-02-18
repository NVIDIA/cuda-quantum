/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"

#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cstdint>

namespace cudaq::realtime {

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
///
/// When KernelType::is_cooperative is true, the kernel is launched via
/// cudaLaunchCooperativeKernel and ALL threads participate in calling the
/// RPC handler (needed for multi-block cooperative decode kernels like BP).
/// Thread 0 polls/parses the header, broadcasts work via shared memory,
/// then all threads call the handler after a grid.sync().
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

  if constexpr (KernelType::is_cooperative) {
    //==========================================================================
    // Cooperative path: ALL threads call the handler.
    //
    // Work descriptor in shared memory (block 0 broadcasts via grid.sync).
    // Only block 0 needs shared memory for the descriptor; other blocks
    // read the device-memory copies after the grid barrier.
    //==========================================================================
    __shared__ DeviceRPCFunction s_func;
    __shared__ void*             s_arg_buffer;
    __shared__ std::uint8_t*     s_output_buffer;
    __shared__ std::uint32_t     s_arg_len;
    __shared__ std::uint32_t     s_max_result_len;
    __shared__ bool              s_have_work;

    // Device-memory work descriptor visible to all blocks after grid.sync.
    // We use a single set since the cooperative kernel processes one RPC at
    // a time (all threads participate, so no pipelining).
    __device__ static DeviceRPCFunction d_func;
    __device__ static void*             d_arg_buffer;
    __device__ static std::uint8_t*     d_output_buffer;
    __device__ static std::uint32_t     d_arg_len;
    __device__ static std::uint32_t     d_max_result_len;
    __device__ static bool              d_have_work;

    while (!(*shutdown_flag)) {
      // --- Phase 1: Thread 0 polls and parses ---
      if (tid == 0) {
        s_have_work = false;
        std::uint64_t rx_value = rx_flags[current_slot];
        if (rx_value != 0) {
          void* rx_slot = reinterpret_cast<void*>(rx_value);
          RPCHeader* header = static_cast<RPCHeader*>(rx_slot);
          if (header->magic == RPC_MAGIC_REQUEST) {
            const cudaq_function_entry_t* entry = dispatch_lookup_entry(
                header->function_id, function_table, func_count);
            if (entry != nullptr &&
                entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
              std::uint8_t* tx_slot = tx_data + current_slot * tx_stride_sz;

              s_func          = reinterpret_cast<DeviceRPCFunction>(
                  entry->handler.device_fn_ptr);
              s_arg_buffer    = static_cast<void*>(header + 1);
              s_output_buffer = tx_slot + sizeof(RPCResponse);
              s_arg_len       = header->arg_len;
              s_max_result_len = tx_stride_sz - sizeof(RPCResponse);
              s_have_work     = true;

              // Publish to device memory for other blocks
              d_func           = s_func;
              d_arg_buffer     = s_arg_buffer;
              d_output_buffer  = s_output_buffer;
              d_arg_len        = s_arg_len;
              d_max_result_len = s_max_result_len;
              d_have_work      = true;
            }
          }
          if (!s_have_work) {
            // Bad magic or unsupported mode -- discard
            __threadfence_system();
            rx_flags[current_slot] = 0;
          }
        }
      }

      // --- Phase 2: Broadcast to all threads ---
      KernelType::sync();

      // Non-block-0 threads read from device memory
      bool have_work;
      DeviceRPCFunction func;
      void* arg_buffer;
      std::uint8_t* output_buffer;
      std::uint32_t arg_len;
      std::uint32_t max_result_len;
      if (blockIdx.x == 0) {
        have_work      = s_have_work;
        func           = s_func;
        arg_buffer     = s_arg_buffer;
        output_buffer  = s_output_buffer;
        arg_len        = s_arg_len;
        max_result_len = s_max_result_len;
      } else {
        have_work      = d_have_work;
        func           = d_func;
        arg_buffer     = d_arg_buffer;
        output_buffer  = d_output_buffer;
        arg_len        = d_arg_len;
        max_result_len = d_max_result_len;
      }

      // --- Phase 3: ALL threads call the handler ---
      std::uint32_t result_len = 0;
      int status = 0;
      if (have_work) {
        status = func(arg_buffer, output_buffer, arg_len,
                       max_result_len, &result_len);
      }

      // --- Phase 4: Sync, then thread 0 writes response ---
      KernelType::sync();

      if (tid == 0 && have_work) {
        std::uint8_t* tx_slot = tx_data + current_slot * tx_stride_sz;
        RPCResponse* response = reinterpret_cast<RPCResponse*>(tx_slot);
        response->magic = RPC_MAGIC_RESPONSE;
        response->status = status;
        response->result_len = result_len;

        __threadfence_system();
        tx_flags[current_slot] = reinterpret_cast<std::uint64_t>(tx_slot);

        __threadfence_system();
        rx_flags[current_slot] = 0;
        local_packet_count++;
        current_slot = (current_slot + 1) % num_slots;
      }

      // Reset device-memory work flag for next iteration
      if (tid == 0) {
        d_have_work = false;
      }

      KernelType::sync();

      if ((local_packet_count & 0xFF) == 0) {
        __threadfence_system();
      }
    }
  } else {
    //==========================================================================
    // Regular path: only thread 0 calls the handler (unchanged).
    //==========================================================================
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

            // Compute TX slot address from symmetric TX data buffer
            std::uint8_t* tx_slot = tx_data + current_slot * tx_stride_sz;

            // Handler writes results directly to TX slot (after response header)
            std::uint8_t* output_buffer = tx_slot + sizeof(RPCResponse);
            std::uint32_t result_len = 0;
            std::uint32_t max_result_len = tx_stride_sz - sizeof(RPCResponse);
            int status = func(arg_buffer, output_buffer, arg_len,
                              max_result_len, &result_len);

            // Write RPC response header to TX slot
            RPCResponse* response = reinterpret_cast<RPCResponse*>(tx_slot);
            response->magic = RPC_MAGIC_RESPONSE;
            response->status = status;
            response->result_len = result_len;

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
    GraphIOContext* graph_io_ctx,
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

            // Handler writes results directly to TX slot (after response header)
            std::uint8_t* output_buffer = tx_slot + sizeof(RPCResponse);
            std::uint32_t result_len = 0;
            std::uint32_t max_result_len = tx_stride_sz - sizeof(RPCResponse);
            int status = func(arg_buffer, output_buffer, arg_len,
                              max_result_len, &result_len);

            // Write RPC response to TX slot
            RPCResponse* response = reinterpret_cast<RPCResponse*>(tx_slot);
            response->magic = RPC_MAGIC_RESPONSE;
            response->status = status;
            response->result_len = result_len;

            __threadfence_system();
            tx_flags[current_slot] = reinterpret_cast<std::uint64_t>(tx_slot);
          }
#if __CUDA_ARCH__ >= 900
          else if (entry->dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH) {
            // Fill IO context so the graph kernel can read input from
            // rx_slot, write the RPCResponse to tx_slot, and signal
            // completion by setting *tx_flag = tx_flag_value.
            if (graph_io_ctx != nullptr) {
              graph_io_ctx->rx_slot = rx_slot;
              graph_io_ctx->tx_slot = tx_slot;
              graph_io_ctx->tx_flag = &tx_flags[current_slot];
              graph_io_ctx->tx_flag_value =
                  reinterpret_cast<std::uint64_t>(tx_slot);
              graph_io_ctx->tx_stride_sz = tx_stride_sz;
              __threadfence_system();
            }

            // Launch pre-created graph (fire-and-forget is async; the
            // graph kernel is responsible for writing the response and
            // signaling tx_flag when done).
            cudaGraphLaunch(entry->handler.graph_exec,
                            cudaStreamGraphFireAndForget);
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

} // namespace cudaq::realtime

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
      cudaq::realtime::dispatch_kernel_device_call_only<cudaq::realtime::RegularKernel>,
      threads_per_block, 0);
  if (err != cudaSuccess) return err;
  if (out_blocks) *out_blocks = num_blocks;
  return cudaSuccess;
}

extern "C" cudaError_t cudaq_dispatch_kernel_cooperative_query_occupancy(
    int* out_blocks, uint32_t threads_per_block) {
  int num_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks,
      cudaq::realtime::dispatch_kernel_device_call_only<
          cudaq::realtime::CooperativeKernel>,
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
  cudaq::realtime::dispatch_kernel_device_call_only<cudaq::realtime::RegularKernel>
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
          cudaq::realtime::dispatch_kernel_device_call_only<cudaq::realtime::CooperativeKernel>),
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
  cudaq::realtime::GraphIOContext* graph_io_ctx;
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
    void* graph_io_ctx_raw,
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
  ctx->graph_io_ctx =
      static_cast<cudaq::realtime::GraphIOContext*>(graph_io_ctx_raw);
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
      &ctx->graph_io_ctx,
      &ctx->shutdown_flag,
      &ctx->stats,
      &ctx->num_slots
  };
  
  kernel_params.func = reinterpret_cast<void*>(
      cudaq::realtime::dispatch_kernel_with_graph<cudaq::realtime::RegularKernel>);
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
