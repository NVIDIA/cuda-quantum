/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Support/Logging.h"

#ifdef LOGGING_BACKEND_QUILL

#include "quill/Backend.h"
#include "quill/Frontend.h"
#include "quill/core/PatternFormatterOptions.h"
#include "quill/sinks/ConsoleSink.h"
#include "quill/sinks/FileSink.h"
#include "quill/sinks/RotatingFileSink.h"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::synth::logging {

namespace {

std::atomic<bool>                                 g_initialized{false};
std::mutex                                        g_init_mutex;
std::vector<std::shared_ptr<quill::Sink>>         g_sinks;
quill::PatternFormatterOptions                    g_pattern_opts;
quill::LogLevel                                   g_global_level  = quill::LogLevel::Info;
bool                                              g_immediate_flush = false;
bool                                              g_started_backend = false;
std::unordered_map<std::string, quill::LogLevel>  g_logger_level_overrides;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

quill::LogLevel parse_log_level(std::string_view s, quill::LogLevel fallback) noexcept {
  if (s == "TRACE" || s == "TRACE_L1") return quill::LogLevel::TraceL1;
  if (s == "TRACE_L2")                 return quill::LogLevel::TraceL2;
  if (s == "TRACE_L3")                 return quill::LogLevel::TraceL3;
  if (s == "DEBUG")                    return quill::LogLevel::Debug;
  if (s == "INFO")                     return quill::LogLevel::Info;
  if (s == "WARNING" || s == "WARN")   return quill::LogLevel::Warning;
  if (s == "ERROR")                    return quill::LogLevel::Error;
  if (s == "CRITICAL")                 return quill::LogLevel::Critical;
  return fallback;
}

LoggingConfig apply_env_overrides(LoggingConfig cfg) {
  if (const char* lvl = std::getenv("CUDAQ_SYNTH_LOG_LEVEL"))
    cfg.global_level = parse_log_level(lvl, cfg.global_level);

  if (const char* file = std::getenv("CUDAQ_SYNTH_LOG_FILE")) {
    SinkConfig fc;
    fc.type      = SinkType::File;
    fc.file_path = file;
    cfg.sinks.push_back(std::move(fc));
  }

  if (const char* fmt = std::getenv("CUDAQ_SYNTH_LOG_FORMAT"))
    cfg.format_pattern = fmt;

  return cfg;
}

std::shared_ptr<quill::Sink> make_sink(const SinkConfig& sc) {
  switch (sc.type) {
    case SinkType::Console:
      return quill::Frontend::create_or_get_sink<quill::ConsoleSink>("synth_console");

    case SinkType::File:
      return quill::Frontend::create_or_get_sink<quill::FileSink>(
          sc.file_path,
          []() {
            quill::FileSinkConfig cfg;
            cfg.set_open_mode('a');
            cfg.set_filename_append_option(quill::FilenameAppendOption::None);
            return cfg;
          }());

    case SinkType::RotatingFile: {
      std::size_t max_size = sc.max_file_size;
      std::string rot_time = sc.rotation_time;
      return quill::Frontend::create_or_get_sink<quill::RotatingFileSink>(
          sc.file_path,
          [max_size, rot_time]() {
            quill::RotatingFileSinkConfig cfg;
            cfg.set_open_mode('a');
            cfg.set_filename_append_option(quill::FilenameAppendOption::None);
            if (max_size > 0)
              cfg.set_rotation_max_file_size(max_size);
            if (!rot_time.empty())
              cfg.set_rotation_time_daily(rot_time);
            return cfg;
          }());
    }
  }
  // Unreachable — suppress compiler warnings.
  return quill::Frontend::create_or_get_sink<quill::ConsoleSink>("synth_console");
}

quill::Logger* create_logger_impl(const char* name) {
  quill::Logger* l;
  if (g_sinks.size() == 1) {
    l = quill::Frontend::create_or_get_logger(name, g_sinks[0], g_pattern_opts);
  } else {
    l = quill::Frontend::create_or_get_logger(
        name,
        std::vector<std::shared_ptr<quill::Sink>>(g_sinks),
        g_pattern_opts);
  }

  l->set_log_level(g_global_level);

  if (g_immediate_flush)
    l->set_immediate_flush(1);

  // Apply per-logger level override if configured.
  auto it = g_logger_level_overrides.find(name);
  if (it != g_logger_level_overrides.end())
    l->set_log_level(it->second);

  return l;
}

} // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void init(const LoggingConfig& config) {
  std::lock_guard<std::mutex> lock(g_init_mutex);
  if (g_initialized.load(std::memory_order_relaxed))
    return;

  LoggingConfig cfg = apply_env_overrides(config);

  g_global_level           = cfg.global_level;
  g_immediate_flush        = cfg.immediate_flush;
  g_logger_level_overrides = cfg.logger_levels;

  if (cfg.start_backend) {
    quill::Backend::start();
    g_started_backend = true;
  }

  g_pattern_opts = quill::PatternFormatterOptions{
      cfg.format_pattern, cfg.timestamp_format, quill::Timezone::LocalTime};

  for (const auto& sc : cfg.sinks) {
    try {
      g_sinks.push_back(make_sink(sc));
    } catch (const std::exception& ex) {
      std::cerr << "[cudaq::synth] Failed to create log sink: "
                << ex.what() << " -- falling back to console.\n";
      try {
        g_sinks.push_back(
            quill::Frontend::create_or_get_sink<quill::ConsoleSink>(
                "synth_console_fallback"));
      } catch (...) {
      }
    }
  }

  if (g_sinks.empty()) {
    g_sinks.push_back(
        quill::Frontend::create_or_get_sink<quill::ConsoleSink>("synth_console"));
  }

  // Pre-create the default "synth" logger so it is ready before any module
  // code runs.
  create_logger_impl("synth");

  // Publish the initialized state last so logger() callers see a consistent
  // view of g_sinks / g_pattern_opts.
  g_initialized.store(true, std::memory_order_release);
}

void shutdown() {
  if (!g_initialized.load(std::memory_order_acquire))
    return;

  // Flush all registered loggers before stopping the backend.
  for (quill::Logger* l : quill::Frontend::get_all_loggers()) {
    if (l)
      l->flush_log();
  }

  if (g_started_backend)
    quill::Backend::stop();
}

quill::Logger* logger(const char* name) {
  // Auto-initialize with defaults if init() was never called.
  if (!g_initialized.load(std::memory_order_acquire))
    init();

  // Fast path: logger already exists (hash-map lookup inside Quill).
  if (quill::Logger* existing = quill::Frontend::get_logger(name))
    return existing;

  // First-time creation for this logger name.
  return create_logger_impl(name);
}

} // namespace cudaq::synth::logging

#else // LOGGING_BACKEND_QUILL not defined — provide empty no-op stubs.

namespace cudaq::synth::logging {
void init(const LoggingConfig&) {}
void shutdown() {}
} // namespace cudaq::synth::logging

#endif // LOGGING_BACKEND_QUILL
