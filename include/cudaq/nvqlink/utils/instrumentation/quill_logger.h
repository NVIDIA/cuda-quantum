/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file quill_logger.hpp
/// @brief Quill logger backend implementation
///
/// This file should only be included via logger.hpp when
/// LOGGING_BACKEND_QUILL is defined. It implements the unified logging API
/// using Quill v11.0.2.
///
/// Quill provides low-latency asynchronous logging with:
///
///   - Lock-free SPSC queues (~20-30ns hot path overhead)
///   - Asynchronous formatting and I/O on dedicated backend thread
///   - BoundedDropping queue that never blocks the hot path
///   - Compile-time log level filtering via QUILL_ACTIVE_LOG_LEVEL

#include <quill/Backend.h>
#include <quill/Frontend.h>
#include <quill/LogMacros.h>
#include <quill/Logger.h>
#include <quill/sinks/ConsoleSink.h>
#include <quill/std/Vector.h>

// Need profiler.hpp for domain macro definitions (DOMAIN_DAEMON, etc.)
#include "cudaq/nvqlink/utils/instrumentation/domains.h"

namespace cudaq::nvqlink::quill_backend {

//===----------------------------------------------------------------------===//
// Custom Frontend Options (following recommended_usage pattern)
//===----------------------------------------------------------------------===//

/// CustomFrontendOptions optimized for low-latency daemon workloads
struct CustomFrontendOptions {
  /// BoundedDropping: NEVER blocks, drops messages if queue is full
  /// This is critical for hot path safety - we never want logging to stall
  /// the hot path, even under extreme load
  static constexpr quill::QueueType queue_type =
      quill::QueueType::BoundedDropping;

  /// Initial per-thread queue capacity (128KB)
  static constexpr size_t initial_queue_capacity = 131'072;

  /// Retry interval for blocking queues (not used with BoundedDropping)
  static constexpr uint32_t blocking_queue_retry_interval_ns = 800;

  /// Maximum capacity for unbounded queues (not used with BoundedDropping)
  static constexpr size_t unbounded_queue_max_capacity =
      2ull * 1024u * 1024u * 1024u;

  /// Huge pages policy (disabled for now)
  static constexpr quill::HugePagesPolicy huge_pages_policy =
      quill::HugePagesPolicy::Never;
};

//===----------------------------------------------------------------------===//
// Using Declarations
//===----------------------------------------------------------------------===//

using CustomFrontend = quill::FrontendImpl<CustomFrontendOptions>;
using CustomLogger = quill::LoggerImpl<CustomFrontendOptions>;

//===----------------------------------------------------------------------===//
// Domain Logger Handles (defined in quill_logger.cpp)
//===----------------------------------------------------------------------===//

extern CustomLogger *logger_daemon;
extern CustomLogger *logger_dispatcher;
extern CustomLogger *logger_memory;
extern CustomLogger *logger_channel;
extern CustomLogger *logger_user;
extern CustomLogger *logger_gpu;

} // namespace cudaq::nvqlink::quill_backend

//===----------------------------------------------------------------------===//
// Namespace Implementation (forward declared in quill_logger.cpp)
//===----------------------------------------------------------------------===//

namespace cudaq::nvqlink::logger {
void initialize();
void shutdown();
} // namespace cudaq::nvqlink::logger

//===----------------------------------------------------------------------===//
// Preprocessor-based domain resolution
//===----------------------------------------------------------------------===//

/// Helper for token pasting to directly map domain identifiers to logger
/// handles at compile-time (zero-overhead dispatch) Requires two-level
/// expansion: domain macro must expand before token pasting
#define _NVQLINK_LOGGER_IMPL(domain) nvqlink::quill_backend::logger_##domain
#define _NVQLINK_LOGGER(domain) _NVQLINK_LOGGER_IMPL(domain)

//===----------------------------------------------------------------------===//
// Unified API Macros (Quill Implementation)
//===----------------------------------------------------------------------===//

// These macros map to Quill's LOG_* macros while adding domain context.
// Quill will compile out lower-level logs based on QUILL_ACTIVE_LOG_LEVEL.
//
// Note: Includes null-pointer checks for safety during early initialization.

#define NVQLINK_LOG_TRACE(domain, fmt, ...)                                    \
  do {                                                                         \
    auto *_logger_ptr = _NVQLINK_LOGGER(domain);                               \
    if (_logger_ptr) {                                                         \
      LOG_TRACE_L3(_logger_ptr, fmt, ##__VA_ARGS__);                           \
    }                                                                          \
  } while (0)

#define NVQLINK_LOG_DEBUG(domain, fmt, ...)                                    \
  do {                                                                         \
    auto *_logger_ptr = _NVQLINK_LOGGER(domain);                               \
    if (_logger_ptr) {                                                         \
      LOG_DEBUG(_logger_ptr, fmt, ##__VA_ARGS__);                              \
    }                                                                          \
  } while (0)

#define NVQLINK_LOG_INFO(domain, fmt, ...)                                     \
  do {                                                                         \
    auto *_logger_ptr = _NVQLINK_LOGGER(domain);                               \
    if (_logger_ptr) {                                                         \
      LOG_INFO(_logger_ptr, fmt, ##__VA_ARGS__);                               \
    }                                                                          \
  } while (0)

#define NVQLINK_LOG_WARNING(domain, fmt, ...)                                  \
  do {                                                                         \
    auto *_logger_ptr = _NVQLINK_LOGGER(domain);                               \
    if (_logger_ptr) {                                                         \
      LOG_WARNING(_logger_ptr, fmt, ##__VA_ARGS__);                            \
    }                                                                          \
  } while (0)

#define NVQLINK_LOG_ERROR(domain, fmt, ...)                                    \
  do {                                                                         \
    auto *_logger_ptr = _NVQLINK_LOGGER(domain);                               \
    if (_logger_ptr) {                                                         \
      LOG_ERROR(_logger_ptr, fmt, ##__VA_ARGS__);                              \
    }                                                                          \
  } while (0)
