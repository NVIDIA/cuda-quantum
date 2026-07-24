/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
// DebugScope -- structured debug logging shim for the synth library
//===----------------------------------------------------------------------===//
//
// Mirrors the MLIR greedy-rewriter idiom (see
// llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp:400-460
// and the LDBG primitives in llvm/Support/DebugLog.h).
//
// What the macros produce:
//   - Every line is auto-prefixed with `[cudaq-synth:1] ` via
//     llvm::impl::raw_ldbg_ostream.
//   - llvm::ScopedPrinter wraps that prefixed stream and tracks indent.
//   - CUDAQ_SYNTH_OPEN / CUDAQ_SYNTH_OPEN_SUB emit `Name {` or `* Name {`
//   and bump
//     the indent; CUDAQ_SYNTH_CLOSE / CUDAQ_SYNTH_CLOSE_SUCCESS /
//     CUDAQ_SYNTH_CLOSE_FAILURE unindent and emit `}` / `} -> success :
//     reason` / `} -> failure : reason`.
//   - CUDAQ_SYNTH_ACTION emits a one-line `** Verb : details` event marker.
//   - CUDAQ_SYNTH_FENCE emits a `//===---===//` horizontal rule between major
//     iterations (matches the greedy rewriter's `logLineComment`).
//
// Why macros rather than RAII: every macro expands to a self-contained
// LLVM_DEBUG({...}) block (or to a no-op under NDEBUG), and no macro
// declares a variable that other code references. As a result the call
// sites compile cleanly under any combination of NDEBUG / -UNDEBUG /
// debug-flag state. The trade-off is that every CUDAQ_SYNTH_OPEN must be paired
// with a CUDAQ_SYNTH_CLOSE_* on every exit path -- there is no destructor to
// fall back on. The gating is local to each macro invocation, exactly as
// it would be for plain LLVM_DEBUG.
//
// Kept private to cudaq/lib/Synthesis. StreamOps.h includes this header,
// so any .cpp that already pulls in StreamOps.h gets the scope macros for
// free.

namespace cudaq::synth {

#ifndef NDEBUG

//===----------------------------------------------------------------------===//
// Internal helpers (non-NDEBUG)
//===----------------------------------------------------------------------===//

/// Thread-local prefixed raw_ostream wrapping llvm::dbgs(). Every output
/// line gets the `[cudaq-synth:1] ` prefix; multi-line content is
/// re-prefixed on every newline by raw_ldbg_ostream. Mirrors
/// GreedyPatternRewriteDriver.cpp:400.
inline llvm::impl::raw_ldbg_ostream &synth_os() {
  static thread_local llvm::impl::raw_ldbg_ostream stream{"[cudaq-synth:1] ",
                                                          llvm::dbgs()};
  return stream;
}

/// Thread-local ScopedPrinter that wraps the prefixed stream and tracks
/// the open-scope indent depth. Mirrors GreedyPatternRewriteDriver.cpp:403.
inline llvm::ScopedPrinter &synth_logger() {
  static thread_local llvm::ScopedPrinter logger{synth_os()};
  return logger;
}

/// Stand-in for llvm::dbgs() inside synth code: returns the prefixed
/// stream after emitting the current indent. Should be called inside
/// LLVM_DEBUG(...) so the write itself is gated on the runtime debug
/// flag, matching the rest of the synth call sites.
inline llvm::raw_ostream &dbgs() { return synth_logger().startLine(); }

/// Backing functions for the CUDAQ_SYNTH_OPEN / CUDAQ_SYNTH_CLOSE /
/// CUDAQ_SYNTH_FENCE macros. Each macro expands to a single function call so
/// the macro itself stays a single statement.
inline void open_scope(llvm::StringRef name, bool sub) {
  auto &logger = synth_logger();
  logger.startLine();
  if (sub)
    logger.getOStream() << "* ";
  logger.getOStream() << name << " {\n";
  logger.indent();
}

inline void close_scope(llvm::StringRef outcome, llvm::StringRef reason) {
  auto &logger = synth_logger();
  logger.unindent();
  logger.startLine() << '}';
  if (!outcome.empty()) {
    logger.getOStream() << " -> " << outcome;
    if (!reason.empty())
      logger.getOStream() << " : " << reason;
  }
  logger.getOStream() << '\n';
}

inline void fence_line() {
  synth_logger().startLine()
      << "//===-------------------------------------------===//\n";
}

#endif // NDEBUG

} // namespace cudaq::synth

//===----------------------------------------------------------------------===//
// Public macros
//===----------------------------------------------------------------------===//

#ifndef NDEBUG

/// Open a top-level scope: emits `Name {` and bumps the indent. Each
/// CUDAQ_SYNTH_OPEN must be paired with exactly one CUDAQ_SYNTH_CLOSE /
/// CUDAQ_SYNTH_CLOSE_SUCCESS / CUDAQ_SYNTH_CLOSE_FAILURE on every
/// exit path.
#define CUDAQ_SYNTH_OPEN(name)                                                 \
  LLVM_DEBUG(::cudaq::synth::open_scope((name), /*sub=*/false))

/// Open a nested scope: emits `* Name {` and bumps the indent.
#define CUDAQ_SYNTH_OPEN_SUB(name)                                             \
  LLVM_DEBUG(::cudaq::synth::open_scope((name), /*sub=*/true))

/// Close a scope without an outcome -- bare `}`.
#define CUDAQ_SYNTH_CLOSE() LLVM_DEBUG(::cudaq::synth::close_scope({}, {}))

/// Close with success: `} -> success : reason`. `reason` may be any value
/// implicitly convertible to llvm::StringRef (string literal, llvm::Twine
/// expression, std::string, ...). Pass {} or "" to omit the reason and
/// emit `} -> success` alone.
#define CUDAQ_SYNTH_CLOSE_SUCCESS(reason)                                      \
  LLVM_DEBUG(::cudaq::synth::close_scope("success", (reason)))

/// Close with failure: `} -> failure : reason`.
#define CUDAQ_SYNTH_CLOSE_FAILURE(reason)                                      \
  LLVM_DEBUG(::cudaq::synth::close_scope("failure", (reason)))

/// Horizontal rule between major iterations.
#define CUDAQ_SYNTH_FENCE() LLVM_DEBUG(::cudaq::synth::fence_line())

/// One-line action marker `** Verb : details`. The macro yields the
/// prefixed stream already past `** Verb : ` so the caller can chain
/// further `<<` followed by a trailing newline. Self-gated via the
/// LLVM_DEBUG check so it can be used outside an explicit LLVM_DEBUG
/// block.
#define CUDAQ_SYNTH_ACTION(verb)                                               \
  for (bool _synth_act =                                                       \
           ::llvm::DebugFlag && ::llvm::isCurrentDebugType("cudaq-synth");     \
       _synth_act; _synth_act = false)                                         \
  ::cudaq::synth::dbgs() << "** " << (verb) << " : "

#else // NDEBUG

#define CUDAQ_SYNTH_OPEN(name) ((void)0)
#define CUDAQ_SYNTH_OPEN_SUB(name) ((void)0)
#define CUDAQ_SYNTH_CLOSE() ((void)0)
#define CUDAQ_SYNTH_CLOSE_SUCCESS(reason) ((void)0)
#define CUDAQ_SYNTH_CLOSE_FAILURE(reason) ((void)0)
#define CUDAQ_SYNTH_FENCE() ((void)0)
#define CUDAQ_SYNTH_ACTION(verb)                                               \
  for (bool _synth_act = false; _synth_act; _synth_act = false)                \
  ::llvm::nulls()

#endif // NDEBUG
