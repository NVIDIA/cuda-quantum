/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file logger.hpp
/// @brief Unified logging API supporting multiple backends (Quill, NONE)
///
/// This header provides a backend-agnostic logging interface. The backend
/// is selected at compile time via -DLOGGING_BACKEND=<QUILL|NONE>
///
/// All backends share common domain definitions (reusing profiler domains)
/// to ensure consistent logging semantics across the codebase.

namespace cudaq::nvqlink::logger {

//===----------------------------------------------------------------------===//
// Log Levels
//===----------------------------------------------------------------------===//

/// @brief Log levels
///
/// @note These levels are used to filter log messages at compile time. Hot path
/// safe logs are compiled out in production.
enum class Level {
  TRACE,   ///< Per-packet/operation details (hot path safe)
  DEBUG,   ///< Diagnostic information (hot path safe)
  INFO,    ///< Informational messages (control path only)
  WARNING, ///< Recoverable issues (control path only)
  ERROR    ///< Failures requiring attention (control path only)
};

} // namespace cudaq::nvqlink::logger

//===----------------------------------------------------------------------===//
// Backend Selection
//===----------------------------------------------------------------------===//

#if defined(LOGGING_BACKEND_QUILL)
  // Quill backend
#include "cudaq/nvqlink/utils/instrumentation/quill_logger.h"

#else
  // No logging - all macros are no-ops

#define NVQLINK_LOG_TRACE(domain, fmt, ...)
#define NVQLINK_LOG_DEBUG(domain, fmt, ...)
#define NVQLINK_LOG_INFO(domain, fmt, ...)
#define NVQLINK_LOG_WARNING(domain, fmt, ...)
#define NVQLINK_LOG_ERROR(domain, fmt, ...)

namespace cudaq::nvqlink::logger {
inline void initialize() {}
inline void shutdown() {}
} // namespace cudaq::nvqlink::logger

#endif // Backend selection
