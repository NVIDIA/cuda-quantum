/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cudaq::realtime {

/// @brief Regular kernel synchronization using __syncthreads().
///
/// Use this for single-block kernels or when only block-level synchronization
/// is needed. Suitable for simple decode handlers that don't require
/// grid-wide coordination.
struct RegularKernel {
  /// @brief Not a cooperative kernel -- handler is called by thread 0 only.
  static constexpr bool is_cooperative = false;
  /// @brief Synchronize threads within a block.
  __device__ static void sync() { __syncthreads(); }
};

/// @brief Cooperative kernel synchronization using grid.sync().
///
/// Use this for multi-block kernels that need grid-wide synchronization,
/// such as complex decoders with data dependencies across blocks.
/// Requires kernel to be launched with cudaLaunchCooperativeKernel.
struct CooperativeKernel {
  /// @brief Cooperative kernel -- handler is called by ALL threads.
  static constexpr bool is_cooperative = true;
  __device__ static void sync() { cooperative_groups::this_grid().sync(); }
};

} // namespace cudaq::realtime
