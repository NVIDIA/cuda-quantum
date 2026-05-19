/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#ifdef LOGGING_BACKEND_QUILL
#include "quill/Logger.h"
#include "quill/core/LogLevel.h"
#include <unordered_map>
#endif

namespace cudaq::synth {

// ---------------------------------------------------------------------------
// Sink configuration
// ---------------------------------------------------------------------------

enum class SinkType { Console, File, RotatingFile };

/// Configuration for a single log sink.
struct SinkConfig {
  SinkType type = SinkType::Console;
  std::string file_path;
  std::size_t max_file_size = 10u * 1024u * 1024u; // 10 MB (RotatingFile only)
  std::string rotation_time; // e.g. "00:00" for daily rotation
};

// ---------------------------------------------------------------------------
// Logging configuration
// ---------------------------------------------------------------------------

/// Complete configuration for the synthesizer logging infrastructure.
///
/// Passed to `logging::init()` by the embedding application.  Sensible defaults
/// are provided for all fields; the common override path is via `env` vars
/// (CUDAQ_SYNTH_LOG_LEVEL, CUDAQ_SYNTH_LOG_FILE, CUDAQ_SYNTH_LOG_FORMAT) which
/// are applied on top of whatever is passed to `init()`.
struct LoggingConfig {
  /// Whether to call quill::Backend::start() during `init()`.
  /// Set to false if the embedding application has already started the backend.
  bool start_backend = true;

  /// Enable immediate (synchronous) flush after every log statement.
  /// Useful for debugging; incurs significant performance overhead.
  bool immediate_flush = false;

  /// Pattern string passed to quill::PatternFormatterOptions.
  std::string format_pattern =
      "%(time) [%(thread_id)] %(short_source_location:<28) "
      "LOG_%(log_level:<9) %(logger:<16) %(message)";

  /// Timestamp format string (`strftime` + Quill fractional extensions).
  std::string timestamp_format = "%H:%M:%S.%Qus";

  /// Sinks to create.  Defaults to a single console sink.
  std::vector<SinkConfig> sinks = {SinkConfig{}};

#ifdef LOGGING_BACKEND_QUILL
  /// Minimum log level applied to all loggers at `init` time.
  quill::LogLevel global_level = quill::LogLevel::Info;

  /// Per-logger level overrides applied when a logger is first created.
  /// Key: logger name (e.g. "synth.grid"); value: desired LogLevel.
  std::unordered_map<std::string, quill::LogLevel> logger_levels;
#endif
};

// ---------------------------------------------------------------------------
// Logging API
// ---------------------------------------------------------------------------

namespace logging {

/// Initialize the logging infrastructure.
///
/// Must be called before any synthesizer code runs (or at least before the
/// first CUDAQ_SYNTH_LOG_* call, which will auto-initialize with defaults if
/// `init()` was not called explicitly).
///
/// Thread-safe; subsequent calls after the first are no-ops.
void init(const LoggingConfig &config = {});

/// Flush all pending log messages and stop the backend thread (if we started
/// it).  Subsequent logging calls after shutdown() are undefined.
///
/// Note: quill::Backend::start() registers an `atexit()` handler that calls
/// Backend::stop() automatically, so explicit shutdown() is only needed when
/// deterministic flushing before a specific program event is required.
void shutdown();

#ifdef LOGGING_BACKEND_QUILL
/// Return the logger for the given subsystem name, creating it if needed.
///
/// Every CUDAQ_SYNTH_LOG_* macro passes a domain name (e.g. "synth.grid",
/// "synth.diophantine") as the first argument. Loggers share the same
/// sinks and format but can have independent runtime log levels.
///
/// Thread-safe.  Called on every CUDAQ_SYNTH_LOG_* invocation; the fast path is
/// a single hash-map lookup after initialization.
quill::Logger *logger(const char *name);
#endif

} // namespace logging

} // namespace cudaq::synth
