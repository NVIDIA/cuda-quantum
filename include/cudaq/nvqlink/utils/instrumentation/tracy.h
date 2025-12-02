/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file tracy.hpp
/// @brief Tracy profiler backend implementation
///
/// This file should only be included via profiler.hpp when
/// PROFILER_BACKEND_TRACY is defined. It implements the unified profiling API
/// using Tracy profiler.
///
/// Tracy provides real-time visualization and low-overhead profiling with:
/// - Automatic memory tracking
/// - Lock contention analysis
/// - Frame markers for game/graphics applications
/// - Cross-platform support

#include <cstdint>
#include <tracy/Tracy.hpp>

namespace cudaq::nvqlink::profiler {

// Tracy auto-initializes, so these are no-ops
inline void initialize() {}
inline void shutdown() {}

} // namespace cudaq::nvqlink::profiler

//===----------------------------------------------------------------------===//
// Helper Macros
//===----------------------------------------------------------------------===//

/// Compile-time string concatenation for "domain::name" format
/// Uses preprocessor stringification and concatenation
#define _NVQLINK_STRINGIFY(x) #x
#define _NVQLINK_TRACY_NAME_IMPL(domain, name)                                 \
  _NVQLINK_STRINGIFY(domain) "::" name
#define NVQLINK_TRACY_NAME(domain, name) _NVQLINK_TRACY_NAME_IMPL(domain, name)

//===----------------------------------------------------------------------===//
// Unified API Macros (Tracy Implementation)
//===----------------------------------------------------------------------===//

/// Basic scoped range with domain context
#define NVQLINK_TRACE_SCOPE(domain, name)                                            \
  ZoneScopedN(NVQLINK_TRACY_NAME(domain, name))

/// Scoped range with custom color
#define NVQLINK_TRACE_SCOPE_COLOR(domain, name, color)                               \
  ZoneNamedNC(_tracy_zone_##__LINE__, NVQLINK_TRACY_NAME(domain, name), color, \
              true)

/// Hot path instrumentation (green)
#define NVQLINK_TRACE_HOTPATH(domain, name)                                          \
  NVQLINK_TRACE_SCOPE_COLOR(domain, name, nvqlink::profiler::COLOR_HOTPATH)

/// Hot path with payload (Tracy doesn't support payloads in zone names,
/// but we can use TracyMessageL to log the value)
#define NVQLINK_TRACE_HOTPATH_PAYLOAD(domain, name, payload)                         \
  ZoneNamedNC(_tracy_zone_##__LINE__, NVQLINK_TRACY_NAME(domain, name),        \
              nvqlink::profiler::COLOR_HOTPATH, true);                         \
  ZoneValue(_tracy_zone_##__LINE__, payload)

/// Full/control path instrumentation (blue)
#define NVQLINK_TRACE_FULL(domain, name)                                             \
  NVQLINK_TRACE_SCOPE_COLOR(domain, name, nvqlink::profiler::COLOR_FULL)

/// Memory operation instrumentation (orange)
#define NVQLINK_TRACE_MEMORY(name)                                                   \
  ZoneNamedNC(_tracy_zone_mem_##__LINE__, NVQLINK_TRACY_NAME(memory, name),    \
              nvqlink::profiler::COLOR_MEMORY, true)

/// User function instrumentation (yellow)
#define NVQLINK_TRACE_USER_RANGE(name)                                               \
  ZoneNamedNC(_tracy_zone_user_##__LINE__, NVQLINK_TRACY_NAME(user, name),     \
              nvqlink::profiler::COLOR_USER, true)

/// Performance counter/plot
/// Tracy has native plot support for visualizing metrics over time
#define NVQLINK_TRACE_COUNTER(name, value) TracyPlot(name, (int64_t)(value))

/// Point marker
#define NVQLINK_TRACE_MARK(domain, msg) TracyMessageL(msg)

/// Error marker (red color)
#define NVQLINK_TRACE_MARK_ERROR(domain, msg)                                        \
  TracyMessageLC(msg, nvqlink::profiler::COLOR_ERROR)

/// Thread naming for better visualization
#define NVQLINK_TRACE_NAME_THREAD(name) tracy::SetThreadName(name)

//===----------------------------------------------------------------------===//
// Tracy-Specific Features
//===----------------------------------------------------------------------===//

/// Memory allocation tracking
/// Allows Tracy to build memory usage graphs and detect leaks
#define NVQLINK_ALLOC(ptr, size) TracyAlloc(ptr, size)

/// Memory free tracking
#define NVQLINK_FREE(ptr) TracyFree(ptr)

/// Frame boundary marker
/// Useful for graphics/game applications to measure frame time
#define NVQLINK_TRACE_FRAME_MARK FrameMark

//===----------------------------------------------------------------------===//
// Additional Tracy Utilities (optional, not part of unified API)
//===----------------------------------------------------------------------===//

/// Lockable wrapper for mutex contention tracking
/// Usage: NVQLINK_LOCKABLE(std::mutex, my_mutex);
#define NVQLINK_LOCKABLE(type, name) tracy::Lockable<type> name

/// Scoped lock with contention tracking
/// Usage: NVQLINK_LOCK_GUARD(my_mutex);
#define NVQLINK_LOCK_GUARD(lockable)                                           \
  std::lock_guard<decltype(lockable)::LockableBase> _tracy_lock_##__LINE__(    \
      lockable)

/// Manual zone text annotation
/// Adds runtime text to the current zone
#define NVQLINK_ZONE_TEXT(text, len) ZoneText(text, len)

/// Manual zone value annotation
/// Adds numeric value to the current zone
#define NVQLINK_ZONE_VALUE(value) ZoneValue(value)
