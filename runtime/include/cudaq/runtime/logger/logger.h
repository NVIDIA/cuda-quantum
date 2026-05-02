/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/runtime/logger/cudaq_fmt.h"
#include "cudaq/runtime/logger/tracer.h"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace cudaq {

/// @brief Returns true if `tag` is enabled. Tags are only enabled/disabled at
/// program startup.
bool isTimingTagEnabled(int tag);

// Keep all spdlog headers hidden in the implementation file.
namespace details {
// This enum must match spdlog::level enums. This is checked via static_assert
// in logger.cpp.
enum class LogLevel { trace, debug, info, warn, error };
bool should_log(const LogLevel logLevel);
void trace(const std::string_view msg);
void info(const std::string_view msg);
void debug(const std::string_view msg);
void warn(const std::string_view msg);
void error(const std::string_view msg);
std::string pathToFileName(const std::string_view fullFilePath);

void logMessagePacked(LogLevel logLevel, const std::string_view message,
                      const std::span<const cudaq_fmt::FormatArgument> &args,
                      const char *funcName, const char *fileName, int lineNo);

void logWithTimestampPacked(
    const std::string_view message,
    const std::span<const cudaq_fmt::FormatArgument> &args);

template <typename... Args>
std::string formatMessage(const std::string_view message, Args &&...args) {
  return cudaq_fmt::format(message, std::forward<Args>(args)...);
}

template <typename... Args>
void logMessage(LogLevel logLevel, const std::string_view message,
                const char *funcName, const char *fileName, int lineNo,
                Args &&...args) {
  std::array<cudaq_fmt::FormatArgument, sizeof...(Args)> packedArgs{
      cudaq_fmt::FormatArgument(std::forward<Args>(args))...};
  logMessagePacked(logLevel, message,
                   std::span<const cudaq_fmt::FormatArgument>(
                       packedArgs.data(), packedArgs.size()),
                   funcName, fileName, lineNo);
}
} // namespace details

/// These types seek to enable automated injection of the source location of the
/// `cudaq::info()` or `debug()` call. The actual formatting is out-of-line in
/// logger.cpp so callers do not need to parse `fmt` or `chrono` headers.
#define CUDAQ_LOGGER_DEDUCTION_STRUCT(NAME)                                    \
  template <typename... Args>                                                  \
  struct NAME {                                                                \
    NAME(const std::string_view message, Args &&...args,                       \
         const char *funcName = __builtin_FUNCTION(),                          \
         const char *fileName = __builtin_FILE(),                              \
         int lineNo = __builtin_LINE()) {                                      \
      if (details::should_log(details::LogLevel::NAME))                        \
        details::logMessage(details::LogLevel::NAME, message, funcName,        \
                            fileName, lineNo, std::forward<Args>(args)...);    \
    }                                                                          \
  };                                                                           \
  template <typename... Args>                                                  \
  NAME(const std::string_view, Args &&...) -> NAME<Args...>;

CUDAQ_LOGGER_DEDUCTION_STRUCT(info);
CUDAQ_LOGGER_DEDUCTION_STRUCT(warn);
CUDAQ_LOGGER_DEDUCTION_STRUCT(error);

#ifdef CUDAQ_DEBUG
CUDAQ_LOGGER_DEDUCTION_STRUCT(debug);
#else
// Remove cudaq::debug log messages from Release binaries.
template <typename... Args>
void debug(const std::string_view, Args &&...) {}
#endif

/// @brief Log a message with timestamp.
// Note 1: This will always log the message regardless of the logging level.
// Note 2: File and line info is not included in the log line.
template <typename... Args>
void log(const std::string_view message, Args &&...args) {
  std::array<cudaq_fmt::FormatArgument, sizeof...(Args)> packedArgs{
      cudaq_fmt::FormatArgument(std::forward<Args>(args))...};
  details::logWithTimestampPacked(
      message, std::span<const cudaq_fmt::FormatArgument>(packedArgs.data(),
                                                          packedArgs.size()));
}

/// @brief This type is meant to provided quick tracing of function calls.
/// Instantiate at the beginning of a function and when it goes out of scope at
/// function end, it will call to the trace function and report the function
/// name and execution time.
class ScopedTrace {
private:
  SpanHandle handle;

  template <typename... Args>
  static std::string formatArgsMsg(bool tagFound, Args &&...args) {
    if constexpr (sizeof...(Args) == 0) {
      (void)tagFound;
      return {};
    } else {
      std::string argsMsg;
      if (tagFound) {
        // Double-escape: cudaq::log() runs the result through fmt::format
        // a second time, so literal braces must survive both passes.
        argsMsg = " (args = {{{{";
        constexpr std::size_t nArgs = sizeof...(Args);
        for (std::size_t i = 0; i < nArgs; i++)
          argsMsg += (i != nArgs - 1) ? "{}, " : "{}}}}})";
      } else {
        argsMsg = " (args = {{";
        constexpr std::size_t nArgs = sizeof...(Args);
        for (std::size_t i = 0; i < nArgs; i++)
          argsMsg += (i != nArgs - 1) ? "{}, " : "{}}})";
      }
      return details::formatMessage(argsMsg, std::forward<Args>(args)...);
    }
  }

public:
  /// @brief Public constructor with a context and a timing tag.
  template <typename... Args>
  ScopedTrace(TraceContext ctx, const int tag, const std::string &name,
              Args &&...args) {
    const bool tagFound = (tag != 0) && cudaq::isTimingTagEnabled(tag);
    if (!tagFound && !details::should_log(details::LogLevel::trace) &&
        !Tracer::instance().isCaptureEnabled())
      return;
    std::string argsMsg = formatArgsMsg(tagFound, std::forward<Args>(args)...);
    handle = Tracer::instance().beginSpan(ctx, name, tag, argsMsg, "scope");
  }

  /// @brief Public constructor with a context and no timing tag.
  template <typename... Args>
  ScopedTrace(TraceContext ctx, const std::string &name, Args &&...args)
      : ScopedTrace(ctx, /*tag=*/0, name, std::forward<Args>(args)...) {}

  ~ScopedTrace() { Tracer::instance().endSpan(std::move(handle)); }
};
} // namespace cudaq

// The following macros avoid the unnecessary processing cost of argument
// evaluation and string formation until after the log level check is done.
#define CUDAQ_ERROR(...)                                                       \
  do {                                                                         \
    ::cudaq::error(__VA_ARGS__);                                               \
    throw std::runtime_error(__VA_ARGS__);                                     \
  } while (false)

#define CUDAQ_WARN(...)                                                        \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::warn)) {      \
      ::cudaq::warn(__VA_ARGS__);                                              \
    }                                                                          \
  } while (false)

#define CUDAQ_INFO(...)                                                        \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::info)) {      \
      ::cudaq::info(__VA_ARGS__);                                              \
    }                                                                          \
  } while (false)

#ifdef CUDAQ_DEBUG
#define CUDAQ_DBG(...)                                                         \
  do {                                                                         \
    if (::cudaq::details::should_log(::cudaq::details::LogLevel::debug)) {     \
      ::cudaq::debug(__VA_ARGS__);                                             \
    }                                                                          \
  } while (false)
#else
#define CUDAQ_DBG(...)
#endif

#define ScopedTraceWithContext(...)                                            \
  cudaq::ScopedTrace trace(cudaq::TraceContext(__builtin_FUNCTION(),           \
                                               __builtin_FILE(),               \
                                               __builtin_LINE()),              \
                           ##__VA_ARGS__)

// Note from Alex:
// I Want to save the below source for later, we should be able to
// use std::source_location to do the above, but this
// only works on GCC for now....

// template <typename... Args>
// struct info {
//   info(
//       const std::string_view message, Args &&...args,
//       const std::source_location &loc = std::source_location::current()) {
//     auto msg = fmt::format(fmt::runtime(message), args...);
//     std::string name = loc.function_name();
//     auto start = name.find_first_of(" ");
//     name = name.substr(start + 1, name.find_first_of("(") - start - 1);
//     std::filesystem::path file = loc.file_name();
//     msg = "[" + file.filename().string() + ":" + std::to_string(loc.line()) +
//           "] " + msg;
//     details::info(msg);
//   }
// };
