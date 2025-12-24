/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_error.h>
#include <doca_gpunetio_verbs_def.h>

#include <cstddef>
#include <cstdint>

namespace cudaq::nvqlink {

// Forward declaration
struct GPUFunctionRegistry;

extern "C" {

/// @brief Prepare receive and send WQEs before starting main loop
///
/// Must be called once before launching the main RPC kernel to initialize
/// WQE entries and post initial receive work requests.
///
/// @param stream CUDA stream for async execution
/// @param qp GPU device queue pair handle
/// @param max_rpc_size Maximum size of RPC request/response
/// @param buffer_mkey Memory key for RDMA buffer
/// @param num_wqes Number of WQEs to prepare (should match WQE_NUM)
/// @return DOCA_SUCCESS or error code
doca_error_t doca_prepare_wqes(cudaStream_t stream, doca_gpu_dev_verbs_qp *qp,
                               std::uint32_t max_rpc_size,
                               std::uint32_t buffer_mkey,
                               std::uint32_t num_wqes);

/// @brief Main RPC processing kernel launcher
///
/// Launches a persistent kernel that:
/// 1. Polls CQ for incoming RPC requests
/// 2. Dispatches to registered GPU functions
/// 3. Sends responses back
/// 4. Reposts receive WQEs
///
/// The kernel runs until exit_flag is set to non-zero.
///
/// @param stream CUDA stream for kernel launch
/// @param qp GPU device queue pair handle
/// @param exit_flag GPU-accessible flag for graceful shutdown
/// @param buffer GPU buffer for RPC data
/// @param page_size Size of each buffer page
/// @param buffer_mkey Memory key for buffer
/// @param num_pages Number of buffer pages
/// @param registry Device-side function registry
/// @param cuda_blocks Number of thread blocks
/// @param cuda_threads Number of threads per block
/// @return DOCA_SUCCESS or error code
///
doca_error_t doca_rpc_kernel(cudaStream_t stream, doca_gpu_dev_verbs_qp *qp,
                             std::uint32_t *exit_flag, std::uint8_t *buffer,
                             std::uint32_t page_size, std::uint32_t buffer_mkey,
                             unsigned num_pages, GPUFunctionRegistry *registry,
                             std::uint32_t cuda_blocks,
                             std::uint32_t cuda_threads);

} // extern "C"

} // namespace cudaq::nvqlink
