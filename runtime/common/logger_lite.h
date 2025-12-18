/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <iostream>
#include <string_view>

namespace cudaq {

// Keep all spdlog headers hidden in the implementation file
namespace details {
// This enum must match spdlog::level enums. This is checked via static_assert
// in Logger.cpp.
enum class LogLevel { trace, debug, info, warn };
bool should_log(const LogLevel logLevel);
void trace(const std::string_view msg);
void info(const std::string_view msg);
void debug(const std::string_view msg);
void warn(const std::string_view msg);
} // namespace details
} // namespace cudaq

#define CUDAQ_WARN_LITE(X)                                                     \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::warn)) {      \
      std::cout << "[WARN] [" << __FILE__ << ":" << __LINE__ << "]";           \
      X;                                                                       \
    }                                                                          \
  } while (false)

#define CUDAQ_INFO_LITE(X)                                                     \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::info)) {      \
      std::cout << "[INFO] [" << __FILE__ << ":" << __LINE__ << "]";           \
      X;                                                                       \
    }                                                                          \
  } while (false)

#define CUDAQ_TRACE_LITE(X)                                                    \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::trace)) {     \
      std::cout << "[TRACE] [" << __FILE__ << ":" << __LINE__ << "]";          \
      X;                                                                       \
    }                                                                          \
  } while (false)

#ifdef CUDAQ_DEBUG
#define CUDAQ_DBG_LITE(X)                                                      \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::debug)) {     \
      std::cout << "[DEBUG] [" << __FILE__ << ":" << __LINE__ << "]";          \
      X;                                                                       \
    }                                                                          \
  } while (false)
#else
#define CUDAQ_DBG_LITE(X)
#endif
