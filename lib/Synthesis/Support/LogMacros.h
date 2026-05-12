/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// ---------------------------------------------------------------------------
// CUDAQ_SYNTH_LOG_* — logging macros for the cudaq::synth module
//
// Every macro requires a subsystem domain name as its first argument.
// This enforces consistent per-subsystem logger assignment at the API level.
//
// Usage:
//   #include "Support/LogMacros.h"
//
//   CUDAQ_SYNTH_LOG_INFO("synth.gridsynth", "theta={:.6f}, eps={:.2e}", theta,
//   eps); CUDAQ_SYNTH_LOG_DEBUG("synth.grid", "solve_tdgp: k={}", k);
//   CUDAQ_SYNTH_LOG_TRACE("synth.diophantine", "candidate residue={}", r);
//
// Guard macros (modeled after LLVM_DEBUG) for code blocks whose sole purpose
// is preparing data for a log call.  Three-layer gating: (1) compiled out
// when LOGGING_BACKEND_QUILL is not defined, (2) dead-code eliminated via
// if constexpr when the compile-time level is above threshold, (3) runtime
// level check via should_log_statement:
//
//   CUDAQ_SYNTH_IF_LOG_TRACE("synth.diophantine", {
//       std::string flist;
//       for (const auto& [p, k] : factors)
//           flist += fmt::format("{}^{} ", p, k);
//       CUDAQ_SYNTH_LOG_TRACE("synth.diophantine", "pending: [{}]", flist);
//   });
//
// Guidelines:
//   CRITICAL / ERROR   Algorithm failures (degenerate inputs, unsolvable
//   cases). WARN               Recoverable issues (timeouts, fallbacks, skipped
//   candidates). INFO               High-level progress (function entry/exit,
//   final T-count). DEBUG              Per-iteration details (loop variables,
//   candidate counts). TRACE              Inner-loop diagnostics.  Always
//   compiled out in release
//                      builds when CUDAQ_SYNTH_LOGGING_LEVEL=INFO or above is
//                      set.
//
//   Never log at INFO or above in tight inner loops.
// ---------------------------------------------------------------------------

#ifdef LOGGING_BACKEND_QUILL

#include "Support/Logging.h"
#include "Support/QuillFormatters.h"
#include "quill/LogMacros.h"

// --- Logging macros (domain name is always the first argument) -------------

#define CUDAQ_SYNTH_LOG_TRACE(name, fmt, ...)                                  \
  LOG_TRACE_L1(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

#define CUDAQ_SYNTH_LOG_DEBUG(name, fmt, ...)                                  \
  LOG_DEBUG(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

#define CUDAQ_SYNTH_LOG_INFO(name, fmt, ...)                                   \
  LOG_INFO(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

#define CUDAQ_SYNTH_LOG_WARN(name, fmt, ...)                                   \
  LOG_WARNING(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

#define CUDAQ_SYNTH_LOG_ERROR(name, fmt, ...)                                  \
  LOG_ERROR(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

#define CUDAQ_SYNTH_LOG_CRITICAL(name, fmt, ...)                               \
  LOG_CRITICAL(::cudaq::synth::logging::logger(name), fmt, ##__VA_ARGS__)

// --- Guard macros (TRACE and DEBUG only) -----------------------------------

#define CUDAQ_SYNTH_IF_LOG_TRACE(name, ...)                                    \
  do {                                                                         \
    if constexpr (static_cast<uint8_t>(quill::LogLevel::TraceL1) >=            \
                  QUILL_COMPILE_ACTIVE_LOG_LEVEL) {                            \
      if (::cudaq::synth::logging::logger(name)                                \
              ->template should_log_statement<quill::LogLevel::TraceL1>()) {   \
        __VA_ARGS__                                                            \
      }                                                                        \
    }                                                                          \
  } while (0)

#define CUDAQ_SYNTH_IF_LOG_DEBUG(name, ...)                                    \
  do {                                                                         \
    if constexpr (static_cast<uint8_t>(quill::LogLevel::Debug) >=              \
                  QUILL_COMPILE_ACTIVE_LOG_LEVEL) {                            \
      if (::cudaq::synth::logging::logger(name)                                \
              ->template should_log_statement<quill::LogLevel::Debug>()) {     \
        __VA_ARGS__                                                            \
      }                                                                        \
    }                                                                          \
  } while (0)

#else // No logging backend — all macros are no-ops that compile away entirely.

#define CUDAQ_SYNTH_LOG_TRACE(name, fmt, ...) ((void)0)
#define CUDAQ_SYNTH_LOG_DEBUG(name, fmt, ...) ((void)0)
#define CUDAQ_SYNTH_LOG_INFO(name, fmt, ...) ((void)0)
#define CUDAQ_SYNTH_LOG_WARN(name, fmt, ...) ((void)0)
#define CUDAQ_SYNTH_LOG_ERROR(name, fmt, ...) ((void)0)
#define CUDAQ_SYNTH_LOG_CRITICAL(name, fmt, ...) ((void)0)

#define CUDAQ_SYNTH_IF_LOG_TRACE(name, ...) ((void)0)
#define CUDAQ_SYNTH_IF_LOG_DEBUG(name, ...) ((void)0)

#endif // LOGGING_BACKEND_QUILL
