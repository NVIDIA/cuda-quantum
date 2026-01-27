/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file dispatch_kernel.cuh
/// @brief Dispatch kernel declarations for external projects.
///
/// The dispatch kernel implementation now lives in a separate CUDA TU
/// (dispatch_kernel.cu) and is linked into libcudaq-realtime.so. This header
/// provides declarations and inline wrappers for the launch functions.

#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaq::nvqlink {

//==============================================================================
// Kernel Launch Function Declarations
//==============================================================================

extern "C" void cudaq_launch_dispatch_kernel_regular(
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
    cudaStream_t stream);

extern "C" void cudaq_launch_dispatch_kernel_cooperative(
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
    cudaStream_t stream);

/// @brief Backward-compatible inline wrapper (regular kernel).
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
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, function_table, function_ids,
      func_count, shutdown_flag, stats, num_slots,
      num_blocks, threads_per_block, stream);
}

/// @brief Backward-compatible inline wrapper (cooperative kernel).
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
  cudaq_launch_dispatch_kernel_cooperative(
      rx_flags, tx_flags, function_table, function_ids,
      func_count, shutdown_flag, stats, num_slots,
      num_blocks, threads_per_block, stream);
}

} // namespace cudaq::nvqlink
