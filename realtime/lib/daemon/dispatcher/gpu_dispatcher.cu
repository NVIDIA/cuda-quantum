/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 *                                                                             *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/gpu_dispatcher.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#ifdef NVQLINK_ENABLE_DOCA
#include "cudaq/nvqlink/network/channels/doca/doca_rpc_kernel.h"
#include "cudaq/nvqlink/daemon/registry/gpu_function_registry.h"
#endif

#include <cstdint>
#include <cuda_runtime.h>

using namespace cudaq::nvqlink;

//==============================================================================
// RPC Protocol Structures
//==============================================================================

/// @brief RPC request header - wire format for function dispatch.
/// Must be wire-compatible with cuda-quantum RPC protocol.
struct __attribute__((packed)) RPCHeader {
  std::uint32_t function_id;  ///< Hash of function name (FNV-1a)
  std::uint32_t arg_len;      ///< Length of argument data in bytes
};

/// @brief RPC response header - returned to caller.
struct __attribute__((packed)) RPCResponse {
  std::int32_t status;        ///< Return status (0 = success)
  std::uint32_t result_len;   ///< Length of result data in bytes
};

//==============================================================================
// Device Function Types
//==============================================================================

/// @brief Device RPC function signature.
/// @param buffer Pointer to argument/result buffer
/// @param arg_len Length of argument data
/// @param max_result_len Maximum result buffer size
/// @param result_len Output: actual result length
/// @return Status code (0 = success)
using DeviceRPCFunction = int (*)(void *buffer, std::uint32_t arg_len,
                                  std::uint32_t max_result_len, 
                                  std::uint32_t *result_len);

//==============================================================================
// Hololink-Style Ring Buffer Interface
//==============================================================================
// 
// The ring buffer uses cudaHostAllocMapped pinned memory with uint64_t flags:
//   - Flag value 0: slot is empty (ready for CPU to write)
//   - Flag value non-zero: contains pointer to data buffer
//
// RX Ring Buffer (CPU writes, GPU reads):
//   rx_flag[i] = 0        -> slot empty
//   rx_flag[i] = data_ptr -> data available at data_ptr
//
// TX Ring Buffer (GPU writes, CPU reads):  
//   tx_flag[i] = 0        -> slot empty
//   tx_flag[i] = data_ptr -> result available at data_ptr
//

/// @brief Context for ring buffer communication.
struct RingBufferContext {
  volatile std::uint64_t* rx_flags;     ///< RX ring buffer flags (CPU->GPU)
  volatile std::uint64_t* tx_flags;     ///< TX ring buffer flags (GPU->CPU)
  std::size_t num_slots;                ///< Number of slots in each ring
  std::size_t slot_size;                ///< Size of each data slot
};

//==============================================================================
// Dispatch Kernel Implementation
//==============================================================================

/// @brief Lookup function in table by function_id.
/// @param function_id Hash of function name
/// @param function_table Array of function pointers
/// @param function_ids Array of function IDs
/// @param func_count Number of functions in table
/// @return Function pointer or nullptr if not found
__device__ DeviceRPCFunction lookup_function(
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

/// @brief Templated dispatch kernel with configurable synchronization and dispatch mode.
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
__global__ void dispatch_kernel_impl(
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
        std::uint32_t function_id = header->function_id;
        std::uint32_t arg_len = header->arg_len;
        
        // Get argument data (immediately after header)
        void* arg_buffer = static_cast<void*>(header + 1);
        
        // Lookup function
        DeviceRPCFunction func = lookup_function(
            function_id, function_table, function_ids, func_count);
        
        if (func != nullptr) {
          // Call the function
          std::uint32_t result_len = 0;
          std::uint32_t max_result_len = 1024; // TODO: Make configurable
          
          int status = func(arg_buffer, arg_len, max_result_len, &result_len);
          
          // Write response header
          RPCResponse* response = static_cast<RPCResponse*>(data_buffer);
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
// Legacy Kernel (for backward compatibility)
//==============================================================================

/// @brief Legacy persistent kernel implementation (PoC stub).
/// This simulates packet processing without actual ring buffer polling.
/// 
/// NOTE: This is a Proof-of-Concept stub. In production, this kernel would:
/// - Run an infinite loop: while(!*shutdown_flag)
/// - Call backend->poll_rx_queue() to receive actual packets
/// - Parse RPCHeader and dispatch to registered handlers
/// - Write responses to tx_queue
__global__ void
daemon_persistent_kernel(void *rx_queue, void *tx_queue, void *buffer_pool,
                         void **function_table, std::uint32_t *function_ids,
                         std::size_t func_count, volatile int *shutdown_flag,
                         std::uint64_t *stats) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::uint64_t local_packet_count = 0;

  // For PoC: Don't run infinite loop, just do limited iterations
  // In production: this would be while(!*shutdown_flag)
  const std::uint64_t MAX_ITERATIONS = 100000; // Limit iterations for PoC

  for (std::uint64_t iter = 0; iter < MAX_ITERATIONS && !(*shutdown_flag);
       iter++) {
    // In production: call backend->poll_rx_queue()
    // For PoC: simulate packet processing

    local_packet_count++;

    // Periodic cooperative yielding
    if (iter % 1000 == 0) {
      __threadfence_system(); // Ensure memory visibility
    }
  }

  // Write stats back (only thread 0)
  if (tid == 0) {
    atomicAdd(reinterpret_cast<unsigned long long*>(stats), local_packet_count);
  }
}

//==============================================================================
// Kernel Launch Functions
//==============================================================================

/// @brief Launch the dispatch kernel with RegularKernel + DeviceCallMode.
void launch_dispatch_kernel_regular(
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
  
  dispatch_kernel_impl<cudaq::realtime::RegularKernel, 
                       cudaq::realtime::DeviceCallMode>
      <<<num_blocks, threads_per_block, 0, stream>>>(
          rx_flags, tx_flags, function_table, function_ids,
          func_count, shutdown_flag, stats, num_slots);
}

/// @brief Launch the dispatch kernel with CooperativeKernel + DeviceCallMode.
void launch_dispatch_kernel_cooperative(
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
          dispatch_kernel_impl<cudaq::realtime::CooperativeKernel,
                               cudaq::realtime::DeviceCallMode>),
      dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, stream);
}

//==============================================================================
// GPUDispatcher Implementation
//==============================================================================

GPUDispatcher::GPUDispatcher(Channel *channel, FunctionRegistry *registry,
                             const ComputeConfig &config)
    : channel_(channel), registry_(registry), config_(config) {}

GPUDispatcher::~GPUDispatcher() {
  stop();

  if (device_shutdown_flag_)
    cudaFree(device_shutdown_flag_);
  if (device_stats_)
    cudaFree(device_stats_);
  if (stream_)
    cudaStreamDestroy(stream_);
}

void GPUDispatcher::start() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUDispatcher::start");
  NVQLINK_TRACE_NAME_THREAD("GPU-Host");

  if (running_.exchange(true)) {
    return;
  }

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Starting GPU dispatcher on device {}",
                   config_.gpu_device_id.value_or(0));

  // Set GPU device
  cudaSetDevice(config_.gpu_device_id.value_or(0));

  // Create stream
  cudaStreamCreate(&stream_);

  // Allocate device memory for control
  cudaMalloc(&device_shutdown_flag_, sizeof(int));
  cudaMalloc(&device_stats_, sizeof(std::uint64_t));

  int zero = 0;
  cudaMemcpy(device_shutdown_flag_, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(device_stats_, 0, sizeof(std::uint64_t));

  // Launch persistent kernel
  launch_persistent_kernel();

  // Check that launch succeeded (non-blocking check)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }

  NVQLINK_LOG_INFO(DOMAIN_GPU,
                   "GPU persistent kernel launched (running asynchronously)");
}

void GPUDispatcher::stop() {
  if (!running_.exchange(false)) {
    return;
  }

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Signaling GPU kernel to shutdown...");

  // Signal kernel to exit (standard RoCE path)
  int shutdown = 1;
  cudaMemcpy(device_shutdown_flag_, &shutdown, sizeof(int),
             cudaMemcpyHostToDevice);

  // For DOCA channels, signal via channel's exit flag
  // This matches Hololink's pattern: cpu_exit_flag[0] = 1
  auto gpu_handles = channel_->get_gpu_memory_handles();
  if (gpu_handles.exit_flag != nullptr) {
    // Direct write to GPU_CPU mapped memory (same as Hololink destructor)
    *gpu_handles.exit_flag = 1;
  }

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Waiting for GPU kernel to complete...");

  // Wait for kernel completion
  // NOTE: DOCA kernel may be blocked in receive() waiting for packets.
  // Like Hololink, we just wait - the kernel exits when a packet arrives
  // and it checks the exit flag, or when the stream is destroyed.

  cudaError_t err = cudaStreamSynchronize(stream_);
  if (err != cudaSuccess) {
    NVQLINK_LOG_WARNING(DOMAIN_DISPATCHER, "Warning: GPU kernel error: {}",
                        cudaGetErrorString(err));
  }

  // Read stats
  std::uint64_t total_packets = 0;
  cudaMemcpy(&total_packets, device_stats_, sizeof(std::uint64_t),
             cudaMemcpyDeviceToHost);

  NVQLINK_LOG_INFO(
      DOMAIN_DISPATCHER,
      "GPU dispatcher stopped. Processed {} iterations (PoC simulation)",
      total_packets);
}

std::uint64_t GPUDispatcher::get_packets_processed() const {
  if (!device_stats_ || !running_.load())
    return 0;

  std::uint64_t packets = 0;
  cudaError_t err = cudaMemcpy(&packets, device_stats_, sizeof(std::uint64_t),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return 0;
  }
  return packets;
}

std::uint64_t GPUDispatcher::get_packets_sent() const {
  return 0;
}

void GPUDispatcher::launch_persistent_kernel() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "launch_persistent_kernel");

  // Get GPU memory handles from backend
  auto gpu_handles = channel_->get_gpu_memory_handles();

  // Get function table from registry
  auto func_table = registry_->get_gpu_function_table();

  // Launch kernel with minimal resources for PoC
  std::uint32_t blocks = config_.gpu_blocks > 0 ? config_.gpu_blocks : 1;
  std::uint32_t threads =
      config_.gpu_threads_per_block > 0 ? config_.gpu_threads_per_block : 32;

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER,
                   "Launching kernel with {} blocks, {} threads per block",
                   blocks, threads);

#ifdef NVQLINK_ENABLE_DOCA
  // Detect DOCA channel via cq_rq_addr (polymorphic detection)
  if (gpu_handles.cq_rq_addr != nullptr) {
    NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Detected DOCA channel, launching DOCA RPC kernel");
    
    auto* gpu_registry = static_cast<GPUFunctionRegistry*>(
        registry_->get_gpu_registry());
    
    doca_error_t result = doca_rpc_kernel(
        stream_,
        static_cast<doca_gpu_dev_verbs_qp*>(gpu_handles.rx_queue_addr),
        gpu_handles.gpu_exit_flag,
        static_cast<std::uint8_t*>(gpu_handles.buffer_pool_addr),
        gpu_handles.page_size,
        gpu_handles.buffer_mkey,
        gpu_handles.num_pages,
        gpu_registry,
        blocks,
        threads);
    
    if (result != DOCA_SUCCESS) {
      throw std::runtime_error("DOCA RPC kernel launch failed");
    }
    return;
  }
#endif

  // Launch standard RoCE kernel ASYNCHRONOUSLY on the stream
  daemon_persistent_kernel<<<blocks, threads, 0, stream_>>>(
      gpu_handles.rx_queue_addr, gpu_handles.tx_queue_addr,
      gpu_handles.buffer_pool_addr, func_table.device_function_ptrs,
      func_table.function_ids, func_table.count,
      static_cast<volatile int*>(device_shutdown_flag_), 
      static_cast<std::uint64_t*>(device_stats_));

  // DON'T synchronize here - let kernel run asynchronously
  // Just check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

void launch_daemon_kernel(void *rx_queue, void *tx_queue, void *buffer_pool,
                          void **function_table, std::uint32_t *function_ids,
                          std::size_t func_count, volatile int *shutdown_flag,
                          std::uint64_t *stats, std::uint32_t num_blocks,
                          std::uint32_t threads_per_block,
                          cudaStream_t stream) {
  daemon_persistent_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      rx_queue, tx_queue, buffer_pool, function_table, function_ids, func_count,
      shutdown_flag, stats);
}
