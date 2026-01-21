/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Logger.h"
#include "FmtCore.h"
#include "Timing.h"
#include "fmt/args.h"
#include <filesystem>
#include <set>
#include <spdlog/cfg/env.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>

namespace cudaq {

// This must be a function rather than a global variable to avoid a startup
// ordering issue that would otherwise occur if we simply made this a global
// variable and then accessed it in the initializeLogger function.
// NOTE: the only time that this list should be modified is at startup time.
static std::set<int> &g_timingList() {
  static std::set<int> timingList;
  return timingList;
}

bool isTimingTagEnabled(int tag) {
  // Note: this function is called very frequently, so it needs to be fast. It
  // is assumed that g_timingList() contains a small number of elements
  // (typically less than 10).
  return g_timingList().contains(tag);
}

/// @brief This function will run at startup and initialize
/// the logger for the runtime to use. It will set the log
/// level and optionally dump to file if specified.
__attribute__((constructor)) void initializeLogger() {
  // Default to no logging
  spdlog::set_level(spdlog::level::warn);

  // but someone can specify CUDAQ_LOG_LEVEL=info (for example)
  // as an environment variable. Can also stack them
  // e.g. CUDAQ_LOG_LEVEL=info,debug
  auto envVal = spdlog::details::os::getenv("CUDAQ_LOG_LEVEL");
  if (!envVal.empty()) {
    spdlog::cfg::helpers::load_levels(envVal);
  }

  envVal = spdlog::details::os::getenv("CUDAQ_LOG_FILE");
  if (!envVal.empty()) {
    auto fileLogger = spdlog::basic_logger_mt("cudaqFileLogger", envVal);
    spdlog::set_default_logger(fileLogger);
    spdlog::flush_on(spdlog::get_level());
  }

  // Parse comma separated integers into g_timingList. Process integer values
  // like this: "1,3,5,7-10,12".
  if (auto *val = std::getenv("CUDAQ_TIMING_TAGS")) {
    std::string valueStr(val);
    std::stringstream ss(valueStr);
    int tag = 0;
    int priorTag = -1; // initialize invalid to invalid tag
    while (ss >> tag) {
      if (tag > cudaq::TIMING_MAX_VALUE)
        fmt::print("WARNING: value in CUDAQ_TIMING_TAGS ({}) is too high and "
                   "will be ignored!\n",
                   tag);
      else
        g_timingList().insert(tag);

      // Handle the A-B range (if necessary)
      if (priorTag != -1)
        for (int t = priorTag + 1; t < tag; t++)
          g_timingList().insert(t);
      if (ss.peek() == ',') {
        priorTag = -1; // this is not a range
        ss.ignore();
      } else if (ss.peek() == '-') {
        priorTag = tag; // save the lower end of the range
        ss.ignore();
      }
    }
  }
}

namespace details {
void trace(const std::string_view msg) { spdlog::trace(msg); }
void info(const std::string_view msg) { spdlog::info(msg); }
void warn(const std::string_view msg) { spdlog::warn(msg); }
void debug(const std::string_view msg) {
#ifdef CUDAQ_DEBUG
  spdlog::debug(msg);
#endif
}
// These asserts are needed for should_log
static_assert(static_cast<int>(LogLevel::debug) ==
                  static_cast<int>(spdlog::level::debug),
              "log level enum mismatch");
static_assert(static_cast<int>(LogLevel::trace) ==
                  static_cast<int>(spdlog::level::trace),
              "log level enum mismatch");
static_assert(static_cast<int>(LogLevel::info) ==
                  static_cast<int>(spdlog::level::info),
              "log level enum mismatch");
static_assert(static_cast<int>(LogLevel::warn) ==
                  static_cast<int>(spdlog::level::warn),
              "log level enum mismatch");
bool should_log(const LogLevel logLevel) {
  return spdlog::should_log(static_cast<spdlog::level::level_enum>(logLevel));
}
std::string pathToFileName(const std::string_view fullFilePath) {
  const std::filesystem::path file(fullFilePath);
  return file.filename().string();
}
} // namespace details
} // namespace cudaq

namespace cudaq_fmt {
namespace details {

void print_packed(const std::string_view message,
                  const std::span<fmt_arg> &args) {
  ::fmt::dynamic_format_arg_store<::fmt::format_context> store;
  for (auto const &a : args)
    std::visit(
        [&](auto const &v) {
          store.push_back(v); // uses the matching fmt::formatter<T>
        },
        a.value);

  if (::fmt::detail::const_check(!::fmt::detail::use_utf8))
    return ::fmt::detail::vprint_mojibake(stdout, message, store, false);
  return ::fmt::vprint_buffered(stdout, message, store);
}

std::string format_packed(const std::string_view fmt_str,
                          const std::span<fmt_arg> &args) {
  ::fmt::dynamic_format_arg_store<::fmt::format_context> store;
  for (auto const &a : args)
    std::visit(
        [&](auto const &v) {
          store.push_back(v); // uses the matching fmt::formatter<T>
        },
        a.value);

  return ::fmt::vformat(fmt_str, store);
}

} // namespace details
} // namespace cudaq_fmt
