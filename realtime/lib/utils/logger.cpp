/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "logger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace cudaq {

__attribute__((constructor)) inline void set_spdlog_handler() {
  static auto logger = [] {
    auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto _logger = std::make_shared<spdlog::logger>("NVQLink Logger", sink);
    _logger->set_level(spdlog::level::trace);
    _logger->set_pattern("[%T] [%^%l%$] %v");
    return _logger;
  }();

  Logger::set_handler([=](Logger::Level level, const std::string &msg) {
    switch (level) {
    case Logger::Level::Info:
      logger->info(msg);
      break;
    case Logger::Level::Debug:
      logger->debug(msg);
      break;
    case Logger::Level::Warn:
      logger->warn(msg);
      break;
    }
  });
}

} // namespace cudaq
