/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>

namespace cudaq::realtime {

/// @brief Device call dispatch mode - direct __device__ function call.
///
/// The handler function is called directly from within the dispatch kernel.
/// This is the simplest and lowest-latency dispatch mode, suitable for
/// lightweight handlers like simple decoders or data transformations.
struct DeviceCallMode {
  /// @brief Dispatch to handler via direct device function call.
  ///
  /// @tparam HandlerFunc Function pointer type
  /// @tparam ContextType Context structure type
  /// @tparam Args Additional argument types
  /// @param handler The __device__ function to call
  /// @param ctx Handler context (matrices, dimensions, etc.)
  /// @param args Additional arguments
  template <typename HandlerFunc, typename ContextType, typename... Args>
  __device__ static void dispatch(HandlerFunc handler, ContextType &ctx,
                                  Args... args) {
    handler(ctx, args...);
  }
};

/// @brief Graph launch dispatch mode - launches a CUDA graph from device.
///
/// The handler is a pre-captured CUDA graph that gets launched from the
/// persistent kernel. This is suitable for complex multi-kernel workflows
/// that benefit from graph optimization.
///
/// NOTE: Requires the graph to be captured and stored in the context at
/// initialization time. The context must contain graph_exec handle.
struct GraphLaunchMode {
  /// @brief Dispatch via CUDA graph launch from device.
  ///
  /// @tparam ContextType Context structure type (must have graph_exec member)
  /// @param ctx Handler context containing the graph executable
  template <typename ContextType>
  __device__ static void dispatch(ContextType &ctx) {
// Device graph launch requires CUDA 12.0+ and appropriate context setup
// The graph_exec must be a cudaGraphExec_t captured at initialization
#if __CUDA_ARCH__ >= 900
    // cudaGraphLaunch is available from device code on Hopper+
    // Note: This is a placeholder - actual implementation requires
    // the graph_exec to be properly set up in the context
    if (ctx.graph_exec != nullptr) {
      cudaGraphLaunch(ctx.graph_exec, ctx.stream);
    }
#endif
  }
};

} // namespace cudaq::realtime
