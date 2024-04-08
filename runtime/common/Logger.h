/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <chrono>

// Be careful about fmt getting into public headers
#include "common/FmtCore.h"

namespace cudaq {

/// @brief Returns true if `tag` is enabled. Tags are only enabled/disabled at
/// program startup.
bool isTimingTagEnabled(int tag);

// Keep all spdlog headers hidden in the implementation file
namespace details {
// This enum must match spdlog::level enums. This is checked via static_assert
// in Logger.cpp.
enum class LogLevel { trace, debug, info };
bool should_log(const LogLevel logLevel);
void trace(const std::string_view msg);
void info(const std::string_view msg);
void debug(const std::string_view msg);
std::string pathToFileName(const std::string_view fullFilePath);
} // namespace details

/// This type seeks to enable automated injection of the
/// source location of the `cudaq::info()` or `debug()` call.
/// We do this via a struct of the same name (info), which
/// takes the message and variadic arguments for the message, and
/// finishes with arguments with defaults providing the file name
/// and the source location. We could also add the function name
/// if we want it in the future.
///
/// We then use a template deduction guide to map calls of the
/// templated `cudaq::info()` function to this type.
#define CUDAQ_LOGGER_DEDUCTION_STRUCT(NAME)                                    \
  template <typename... Args>                                                  \
  struct NAME {                                                                \
    NAME(const std::string_view message, Args &&...args,                       \
         const char *funcName = __builtin_FUNCTION(),                          \
         const char *fileName = __builtin_FILE(),                              \
         int lineNo = __builtin_LINE()) {                                      \
      if (details::should_log(details::LogLevel::NAME)) {                      \
        auto msg = fmt::format(fmt::runtime(message), args...);                \
        std::string name = funcName;                                           \
        auto start = name.find_first_of(" ");                                  \
        name = name.substr(start + 1, name.find_first_of("(") - start - 1);    \
        msg = "[" + details::pathToFileName(fileName) + ":" +                  \
              std::to_string(lineNo) + "] " + msg;                             \
        details::NAME(msg);                                                    \
      }                                                                        \
    }                                                                          \
  };                                                                           \
  template <typename... Args>                                                  \
  NAME(const std::string_view, Args &&...) -> NAME<Args...>;

CUDAQ_LOGGER_DEDUCTION_STRUCT(info);

#ifdef CUDAQ_DEBUG
CUDAQ_LOGGER_DEDUCTION_STRUCT(debug);
#else
// Remove cudaq::debug log messages from Release binaries
template <typename... Args>
void debug(const std::string_view, Args &&...) {}
#endif

/// @brief Log a message with timestamp.
// Note 1: This will always log the message regardless of the logging level.
// Note 2: File and line info is not included in the log line.
template <typename... Args>
void log(const std::string_view message, Args &&...args) {
  const auto timestamp = std::chrono::system_clock::now();
  const auto now_c = std::chrono::system_clock::to_time_t(timestamp);
  std::tm now_tm = *std::localtime(&now_c);
  fmt::print("[{:04}-{:02}-{:02} {:02}:{:02}:{:%S}] {}\n",
             now_tm.tm_year + 1900, now_tm.tm_mon + 1, now_tm.tm_mday,
             now_tm.tm_hour, now_tm.tm_min,
             std::chrono::round<std::chrono::milliseconds>(
                 timestamp.time_since_epoch()),
             fmt::format(fmt::runtime(message), args...));
}

/// @brief Context information (function, file, and line) of a caller
struct TraceContext {
  const char *funcName = nullptr;
  const char *fileName = nullptr;
  int lineNo = 0;

  TraceContext(const char *func = __builtin_FUNCTION(),
               const char *file = __builtin_FILE(), int line = __builtin_LINE())
      : funcName(func), fileName(file), lineNo(line) {}
};

/// @brief This type is meant to provided quick tracing
/// of function calls. Instantiate at the beginning
/// of a function and when it goes out of scope at function
/// end, it will call to the trace function and report
/// the function name and the execution time in ms.
//
// Since this traces upon destruction, tracing in tandem
// with ScopeTrace instances within functions called from
// the current one will stack and can be read in reverse order.
// This type keeps track of an integer that prints and indentation
// before the message to demonstrate the call stack.
//
// e.g. for
// void bar() {
//   ScopedTrace trace("bar");
//   foobar() // <-- also creates a ScopedTrace
// }
// void foo() {
//  ScopedTrace trace("foo");
//  bar()
// }
//
// will print
// [2022-12-15 18:54:39.346] [trace] -- foobar() executed in 0.026 ms.
// [2022-12-15 18:54:39.347] [trace] - bar executed in 0.604 ms.
// [2022-12-15 18:54:39.347] [trace] foo executed in 2.572 ms.
class ScopedTrace {
private:
  /// @brief Time when this ScopedTrace is created.
  std::chrono::time_point<std::chrono::system_clock> startTime;

  /// @brief The name of this trace, typically the function name
  std::string traceName;

  /// @brief Any arguments the user would also like to print
  std::string argsMsg;

  /// @brief Integer timing tag value (used to enable/disable this trace)
  int tag = 0;

  /// @brief Whether or not timing tag is enabled
  bool tagFound = false;

  /// @brief File, line, etc. of trace caller
  TraceContext context;

  thread_local static inline short int globalTraceStack = -1;

  /// @brief Constructor with name only. This is private because you should
  /// probably be using ScopedTraceWithContext() instead.
  ScopedTrace(const std::string &name) {
    if (details::should_log(details::LogLevel::trace)) {
      startTime = std::chrono::system_clock::now();
      traceName = name;
      globalTraceStack++;
    }
  }

  /// @brief Constructor, take and print user-specified critical arguments. This
  /// is private because you should probably be using ScopedTraceWithContext()
  /// instead.
  template <typename... Args>
  ScopedTrace(const std::string &name, Args &&...args) {
    if (details::should_log(details::LogLevel::trace)) {
      startTime = std::chrono::system_clock::now();
      traceName = name;
      argsMsg = " (args = {{";
      constexpr std::size_t nArgs = sizeof...(Args);
      for (std::size_t i = 0; i < nArgs; i++) {
        argsMsg += (i != nArgs - 1) ? "{}, " : "{}}})";
      }
      argsMsg = fmt::format(fmt::runtime(argsMsg), args...);
      globalTraceStack++;
    }
  }

  /// @brief Constructor, take and print user-specified critical arguments. This
  /// is private because you should probably be using ScopedTraceWithContext()
  /// instead.
  /// @param tag See Timing.h
  /// @param name String to print
  template <typename... Args>
  ScopedTrace(const int tag, const std::string &name, Args &&...args)
      : tag(tag) {
    tagFound = cudaq::isTimingTagEnabled(tag);
    if (tagFound || details::should_log(details::LogLevel::trace)) {
      startTime = std::chrono::system_clock::now();
      traceName = name;
      if (tagFound) {
        // This needs double double braces because it goes through
        // fmt::format(fmt::runtime()) twice ... once in this function and once
        // in the cudaq::log() in the destructor.
        argsMsg = " (args = {{{{";
        constexpr std::size_t nArgs = sizeof...(Args);
        for (std::size_t i = 0; i < nArgs; i++) {
          argsMsg += (i != nArgs - 1) ? "{}, " : "{}}}}})";
        }
      } else {
        argsMsg = " (args = {{";
        constexpr std::size_t nArgs = sizeof...(Args);
        for (std::size_t i = 0; i < nArgs; i++) {
          argsMsg += (i != nArgs - 1) ? "{}, " : "{}}})";
        }
      }
      argsMsg = fmt::format(fmt::runtime(argsMsg), args...);
      globalTraceStack++;
    }
  }

  /// @brief The constructor with a timing tag. This is private because you
  /// should probably be using ScopedTraceWithContext() instead.
  /// @param tag See Timing.h
  /// @param name String to print
  ScopedTrace(const int tag, const std::string &name,
              const char *funcName = __builtin_FUNCTION(),
              const char *fileName = __builtin_FILE(),
              int lineNo = __builtin_LINE())
      : tag(tag), context(funcName, fileName, lineNo) {
    tagFound = cudaq::isTimingTagEnabled(tag);
    if (tagFound || details::should_log(details::LogLevel::trace)) {
      startTime = std::chrono::system_clock::now();
      traceName = name;
      globalTraceStack++;
    }
  }

public:
  /// @brief Public constructor with a context and a timing tag.
  template <typename... Args>
  ScopedTrace(TraceContext ctx, const int tag, const std::string &name,
              Args &&...args)
      : ScopedTrace(tag, name, args...) {
    context = ctx;
  }

  /// @brief Public constructor with a context and no timing tag.
  template <typename... Args>
  ScopedTrace(TraceContext ctx, const std::string &name, Args &&...args)
      : ScopedTrace(name, args...) {
    context = ctx;
  }

  /// The destructor, get the elapsed time and trace.
  ~ScopedTrace() {
    if (tagFound || details::should_log(details::LogLevel::trace)) {
      auto duration = static_cast<double>(
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::system_clock::now() - startTime)
              .count() /
          1000.0);
      // If we're printing because the tag was found, then add that tag info
      std::string tagStr = tagFound ? fmt::format("[tag={}] ", tag) : "";
      std::string sourceInfo =
          context.fileName
              ? fmt::format("[{}:{}] ",
                            details::pathToFileName(context.fileName),
                            context.lineNo)
              : "";
      auto str = fmt::format(
          "{}{}{}{} executed in {} ms.{}",
          globalTraceStack > 0 ? std::string(globalTraceStack, '-') + " " : "",
          tagStr, sourceInfo, traceName, duration, argsMsg);
      if (tagFound)
        cudaq::log(str);
      else
        details::trace(str);
      globalTraceStack--;
    }
  }
};
} // namespace cudaq

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
