/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <filesystem>

// Be careful about fmt getting into public headers
#include "common/FmtCore.h"

namespace cudaq {

// Keep all spdlog headers hidden in the implementation file
namespace details {
void trace(const std::string_view msg);
void info(const std::string_view msg);
void debug(const std::string_view msg);
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
      auto msg = fmt::format(fmt::runtime(message), args...);                  \
      std::string name = funcName;                                             \
      auto start = name.find_first_of(" ");                                    \
      name = name.substr(start + 1, name.find_first_of("(") - start - 1);      \
      std::filesystem::path file = fileName;                                   \
      msg = "[" + file.filename().string() + ":" + std::to_string(lineNo) +    \
            "] " + msg;                                                        \
      details::NAME(msg);                                                      \
    }                                                                          \
  };                                                                           \
  template <typename... Args>                                                  \
  NAME(const std::string_view, Args &&...) -> NAME<Args...>;

CUDAQ_LOGGER_DEDUCTION_STRUCT(info);
CUDAQ_LOGGER_DEDUCTION_STRUCT(debug);

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

  static inline short int globalTraceStack = -1;

public:
  /// @brief The constructor
  ScopedTrace(const std::string &name)
      : startTime(std::chrono::system_clock::now()), traceName(name) {
    globalTraceStack++;
  }

  /// @brief  Constructor, take and print user-specified critical arguments
  template <typename... Args>
  ScopedTrace(const std::string &name, Args &&...args)
      : startTime(std::chrono::system_clock::now()), traceName(name) {
    argsMsg = " (args = {{";
    constexpr std::size_t nArgs = sizeof...(Args);
    for (std::size_t i = 0; i < nArgs; i++) {
      argsMsg += (i != nArgs - 1) ? "{}, " : "{}}})";
    }
    argsMsg = fmt::format(fmt::runtime(argsMsg), args...);
    globalTraceStack++;
  }

  /// The destructor, get the elapsed time and trace.
  ~ScopedTrace() {
    auto duration = static_cast<double>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now() - startTime)
            .count() /
        1000.0);
    details::trace(fmt::format(
        "{}{} executed in {} ms.{}",
        globalTraceStack > 0 ? std::string(globalTraceStack, '-') + " " : "",
        traceName, duration, argsMsg));
    globalTraceStack--;
  }
};
} // namespace cudaq

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
