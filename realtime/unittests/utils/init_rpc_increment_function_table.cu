/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file init_rpc_increment_function_table.cu
/// @brief Device-side increment RPC handler and function table initialisation.
///
/// This file is compiled by nvcc so that the __device__ function pointer
/// can be taken.  The host-callable setup_rpc_increment_function_table()
/// wrapper is extern "C" so that the bridge .cpp (compiled by g++) can
/// call it without needing CUDA kernel launch syntax.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"

#include <cstdint>
#include <cstring> // std::memset
#include <cuda_runtime.h>
#include <stdexcept>

namespace {

//==============================================================================
// Increment RPC Handler
//==============================================================================

/// @brief Simple RPC handler that increments each byte of the payload by 1.
///
/// Matches the DeviceRPCFunction signature.  Reads from input, writes to
/// output (no in-place overlap).
__device__ int rpc_increment_handler(const void *input, void *output,
                                     std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t *result_len) {

  const std::uint8_t *in_data = static_cast<const std::uint8_t *>(input);
  std::uint8_t *out_data = static_cast<std::uint8_t *>(output);
  std::uint32_t len = (arg_len < max_result_len) ? arg_len : max_result_len;
  for (std::uint32_t i = 0; i < len; ++i)
    out_data[i] = static_cast<std::uint8_t>(in_data[i] + 1);
  *result_len = len;
  return 0;
}

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");

/// @brief Kernel to populate a cudaq_function_entry_t with the increment
///        handler.
__global__ void init_function_table_kernel(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&rpc_increment_handler);
    entries[0].function_id = RPC_INCREMENT_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Schema: 1 array argument (uint8), 1 array result (uint8)
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 0;
    entries[0].schema.args[0].num_elements = 0;
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 0;
    entries[0].schema.results[0].num_elements = 0;
  }
}

//==============================================================================
// Increment RPC Graph Handler
//==============================================================================

__global__ void graph_increment_kernel(void **mailbox_slot_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    auto *io_ctx =
        reinterpret_cast<cudaq::realtime::GraphIOContext *>(*mailbox_slot_ptr);
    if (!io_ctx || !io_ctx->rx_slot)
      return;

    auto *header = static_cast<cudaq::realtime::RPCHeader *>(io_ctx->rx_slot);
    std::uint8_t *measurements = reinterpret_cast<std::uint8_t *>(header + 1);

    auto *response =
        reinterpret_cast<cudaq::realtime::RPCResponse *>(io_ctx->tx_slot);
    std::uint8_t *result =
        io_ctx->tx_slot + sizeof(cudaq::realtime::RPCResponse);

    for (std::uint32_t i = 0; i < header->arg_len; ++i)
      result[i] = measurements[i] + 1;

    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = header->arg_len;
    response->request_id = header->request_id;
    response->ptp_timestamp = header->ptp_timestamp;

    __threadfence_system();
    if (io_ctx->tx_flag)
      *(io_ctx->tx_flag) = io_ctx->tx_flag_value;
  }
}

/// Creates an executable graph that runs graph_increment_kernel with
/// kernel arg = d_mailbox_bank (device pointer to first mailbox slot).
/// Caller must cudaGraphExecDestroy / cudaGraphDestroy.
bool create_increment_graph(void **d_mailbox_bank, cudaGraph_t *graph_out,
                            cudaGraphExec_t *exec_out) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  // kernelParams[i] must be a *pointer to* the i-th argument value.
  // The kernel takes void** so we pass &d_mailbox_bank (a void***).
  cudaKernelNodeParams params = {};
  void *kernel_args[] = {&d_mailbox_bank};
  params.func = reinterpret_cast<void *>(graph_increment_kernel);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(32, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = kernel_args;
  params.extra = nullptr;

  cudaGraphNode_t node = nullptr;
  if (cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params) !=
      cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  *graph_out = graph;
  *exec_out = exec;
  return true;
}
} // anonymous namespace

//==============================================================================
// Host-Callable Wrapper
//==============================================================================

extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *d_entries) {
  init_function_table_kernel<<<1, 1>>>(d_entries);
  cudaDeviceSynchronize();
}

extern "C" void setup_rpc_graph_increment_function_table(
    cudaq_function_entry_t *h_entries, void **d_mailbox_bank,
    cudaGraph_t *graph_out, cudaGraphExec_t *exec_out) {
  const bool graph_created =
      create_increment_graph(d_mailbox_bank, graph_out, exec_out);
  if (!graph_created) {
    throw std::runtime_error(
        "Failed to create CUDA graph for the increment function.");
  }

  // --- Function table (one GRAPH_LAUNCH entry) ---
  std::memset(h_entries, 0, sizeof(cudaq_function_entry_t));
  h_entries->function_id = RPC_INCREMENT_FUNCTION_ID;
  h_entries->dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
  h_entries->handler.graph_exec = *exec_out;
}
