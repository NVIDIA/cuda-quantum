/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/utils/instrumentation/domains.h"

#include <cstdint>

/// @file profiler.hpp
/// @brief Unified profiling API supporting multiple backends (NVTX, Tracy)
///
/// This header provides a backend-agnostic profiling interface. The backend
/// is selected at compile time via -DPROFILER_BACKEND=<NVTX|TRACY|NONE>
///
/// All backends share common domain and category definitions to ensure
/// consistent profiling semantics across tools.

namespace cudaq::nvqlink::profiler {

//===----------------------------------------------------------------------===//
// Shared Constants (used by all backends)
//===----------------------------------------------------------------------===//

/// Domain identifiers for logical component grouping
/// These are macro identifiers for zero-overhead preprocessor-based dispatch
/// Usage: NVQLINK_TRACE_FULL(DOMAIN_DAEMON, "operation")

constexpr const char *DOMAIN_DAEMON_STR = "daemon";
constexpr const char *DOMAIN_DISPATCHER_STR = "dispatcher";
constexpr const char *DOMAIN_MEMORY_STR = "memory";
constexpr const char *DOMAIN_CHANNEL_STR = "channel";
constexpr const char *DOMAIN_USER_STR = "user";
constexpr const char *DOMAIN_GPU_STR = "gpu";

//===----------------------------------------------------------------------===//
// Category identifiers for operation classification
//===----------------------------------------------------------------------===//

/// Datapath operations (critical performance)
constexpr uint32_t CAT_HOT_PATH = 1;

/// Control path operations (initialization, etc.)
constexpr uint32_t CAT_FULL = 2;

/// Performance counters and metrics
constexpr uint32_t CAT_METRICS = 3;

//===----------------------------------------------------------------------===//
// Standard colors (ARGB format) for visual consistency
//===----------------------------------------------------------------------===//

constexpr uint32_t COLOR_HOTPATH = 0xFF00FF00; ///< Green - hot path operations
constexpr uint32_t COLOR_FULL = 0xFF0000FF; ///< Blue - control/init operations
constexpr uint32_t COLOR_USER = 0xFFFFFF00; ///< Yellow - user functions
constexpr uint32_t COLOR_MEMORY = 0xFFFF8000;  ///< Orange - memory operations
constexpr uint32_t COLOR_ERROR = 0xFFFF0000;   ///< Red - errors
constexpr uint32_t COLOR_METRICS = 0xFF00FFFF; ///< Cyan - metrics/counters

} // namespace cudaq::nvqlink::profiler

//===----------------------------------------------------------------------===//
// Backend Selection
//===----------------------------------------------------------------------===//

#if defined(PROFILER_BACKEND_NVTX)
  // NVTX/Nsight backend
#include "nvqlink/instrumentation/nvtx.h"

#elif defined(PROFILER_BACKEND_TRACY)
  // Tracy profiler backend
#include "nvqlink/instrumentation/tracy.h"

#else
  // No profiling - all macros are no-ops

#define NVQLINK_TRACE_SCOPE(domain, name)
#define NVQLINK_TRACE_SCOPE_COLOR(domain, name, color)
#define NVQLINK_TRACE_HOTPATH(domain, name)
#define NVQLINK_TRACE_HOTPATH_PAYLOAD(domain, name, payload)
#define NVQLINK_TRACE_FULL(domain, name)
#define NVQLINK_TRACE_MEMORY(name)
#define NVQLINK_TRACE_USER_RANGE(name)
#define NVQLINK_TRACE_COUNTER(name, value)
#define NVQLINK_TRACE_MARK(domain, msg)
#define NVQLINK_TRACE_MARK_ERROR(domain, msg)
#define NVQLINK_TRACE_NAME_THREAD(name)

#define NVQLINK_ALLOC(ptr, size)
#define NVQLINK_FREE(ptr)
#define NVQLINK_TRACE_FRAME_MARK

namespace cudaq::nvqlink::profiler {
inline void initialize() {}
inline void shutdown() {}
} // namespace cudaq::nvqlink::profiler

#endif // Backend selection
