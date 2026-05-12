/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
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

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace cudaq::realtime {

//==============================================================================
// Kernel Launch Function Declarations (with schema-driven function table)
//==============================================================================
// These declarations match the extern "C" functions defined in dispatch_kernel.cu
// and cudaq_realtime.h

/// @brief Inline wrapper for regular kernel (schema-aware).
inline void launch_dispatch_kernel_regular_inline(
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
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data,
      rx_stride_sz, tx_stride_sz,
      function_table, func_count,
      shutdown_flag, stats, num_slots,
      num_blocks, threads_per_block, stream);
}

/// @brief Inline wrapper for cooperative kernel (schema-aware).
inline void launch_dispatch_kernel_cooperative_inline(
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
  cudaq_launch_dispatch_kernel_cooperative(
      rx_flags, tx_flags, rx_data, tx_data,
      rx_stride_sz, tx_stride_sz,
      function_table, func_count,
      shutdown_flag, stats, num_slots,
      num_blocks, threads_per_block, stream);
}

} // namespace cudaq::realtime
