/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Logger.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <spdlog/cfg/env.h>
#include <spdlog/cfg/helpers.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/spdlog.h>

namespace cudaq {
/// @brief This function will run at startup and initialize
/// the logger for the runtime to use. It will set the log
/// level and optionally dump to file if specified.
__attribute__((constructor)) void initializeLogger() {
  // Default to no logging
  spdlog::set_level(spdlog::level::off);

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
}

namespace details {
void trace(const std::string_view msg) { spdlog::trace(msg); }
void info(const std::string_view msg) { spdlog::info(msg); }
void debug(const std::string_view msg) {
#ifdef CUDAQ_DEBUG
  spdlog::debug(msg);
#endif
}
} // namespace details
} // namespace cudaq
