/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// ---------------------------------------------------------------------------
// DebugScope.h -- structured debug logging shim for the synth library.
//
// Mirrors the MLIR greedy-rewriter idiom (see
// llvm-project/mlir/lib/Transforms/Utils/GreedyPatternRewriteDriver.cpp:400-460
// and the LDBG primitives in llvm/Support/DebugLog.h):
//
//   - Every line is auto-prefixed with `[cudaq-synth:1] ` via
//     llvm::impl::raw_ldbg_ostream.
//   - llvm::ScopedPrinter wraps that prefixed stream and tracks indent.
//   - SYNTH_OPEN / SYNTH_OPEN_SUB emit `Name {` / `* Name {` and bump
//     indent; SYNTH_CLOSE / SYNTH_CLOSE_SUCCESS / SYNTH_CLOSE_FAILURE pop
//     indent and emit `}` / `} -> success : reason` / `} -> failure : reason`.
//   - SYNTH_ACTION emits a one-line `** Verb : details` event marker.
//   - SYNTH_FENCE emits a `//===---===//` horizontal rule between major
//     iterations (matches the greedy rewriter's `logLineComment`).
//
// Every macro expands to a self-contained `LLVM_DEBUG({...})` block (or to
// a no-op under NDEBUG). No macro declares a variable that other code
// references, so the call sites compile cleanly regardless of NDEBUG /
// -UNDEBUG / debug-flag state. Each open *must* be matched by a close on
// every exit path -- there is no RAII to fall back on. The cost is more
// lines at call sites; the benefit is that the gating is local to each
// macro invocation, exactly like LLVM_DEBUG itself.
//
// Kept private to cudaq/lib/Synthesis. Reached transitively via
// Support/StreamOps.h, so any .cpp that already includes StreamOps.h gets
// the scope macros for free.
// ---------------------------------------------------------------------------

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

namespace cudaq::synth {

#ifndef NDEBUG

/// Thread-local llvm::impl::raw_ldbg_ostream wrapping llvm::dbgs() with the
/// `[cudaq-synth:1] ` line prefix. Mirrors
/// GreedyPatternRewriteDriver.cpp:400.
inline llvm::impl::raw_ldbg_ostream &synth_os() {
  static thread_local llvm::impl::raw_ldbg_ostream stream{
      "[cudaq-synth:1] ", llvm::dbgs()};
  return stream;
}

/// Thread-local ScopedPrinter wrapping the prefixed stream. Mirrors
/// GreedyPatternRewriteDriver.cpp:403.
inline llvm::ScopedPrinter &synth_logger() {
  static thread_local llvm::ScopedPrinter logger{synth_os()};
  return logger;
}

/// Replacement for llvm::dbgs() at call sites: returns the prefixed stream
/// after writing the current indent. Multi-line content is automatically
/// re-prefixed on every newline by raw_ldbg_ostream.
///
/// Should be used inside LLVM_DEBUG(...) so the write is itself gated on
/// the runtime debug flag, matching the rest of the synth call sites.
inline llvm::raw_ostream &dbgs() { return synth_logger().startLine(); }

/// Internal helpers backing the SYNTH_OPEN / SYNTH_CLOSE / SYNTH_FENCE
/// macros. They are regular functions so each macro expansion stays a
/// single statement.
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

/// Plain-C++ scope guard for *coroutines* only. Coroutines can be
/// destroyed mid-yield when the consumer abandons the generator; in that
/// case no explicit SYNTH_CLOSE_* in the function body would run, leaving
/// the thread-local ScopedPrinter indent permanently bumped and the next
/// call's output mis-indented. The guard's destructor closes the scope
/// (and pops the indent) on any path out of the coroutine.
///
/// The guard is plain user-visible C++ (no macro hides the declaration),
/// so the variable is always present regardless of NDEBUG / debug-flag
/// state. The class is defined under both NDEBUG and non-NDEBUG below;
/// under NDEBUG the dtor and setters compile to no-ops.
///
/// Usage in a coroutine:
///   generator<X> my_coro(...) {
///     SYNTH_OPEN_SUB("my_coro");
///     cudaq::synth::CloseGuard guard;
///     ...
///     for (...) {
///       co_yield ...;
///     }
///     guard.succeed("yielded N");
///   }
class CloseGuard {
public:
  CloseGuard() = default;
  CloseGuard(const CloseGuard &) = delete;
  CloseGuard &operator=(const CloseGuard &) = delete;
  ~CloseGuard() { close_scope(outcome_, reason_); }

  void succeed(llvm::StringRef reason = {}) {
    outcome_ = "success";
    reason_ = reason.str();
  }
  void fail(llvm::StringRef reason = {}) {
    outcome_ = "failure";
    reason_ = reason.str();
  }

private:
  llvm::StringRef outcome_;
  std::string reason_;
};

#else // NDEBUG

/// No-op stand-in so `cudaq::synth::CloseGuard guard;` is always a valid
/// declaration regardless of build mode.
class CloseGuard {
public:
  void succeed(llvm::StringRef = {}) {}
  void fail(llvm::StringRef = {}) {}
};

#endif // NDEBUG

} // namespace cudaq::synth

#ifndef NDEBUG

/// Open a top-level scope: emits `Name {` and bumps indent. Pair with
/// exactly one SYNTH_CLOSE / SYNTH_CLOSE_SUCCESS / SYNTH_CLOSE_FAILURE on
/// every exit path.
#define SYNTH_OPEN(name)                                                       \
  LLVM_DEBUG(::cudaq::synth::open_scope((name), /*sub=*/false))

/// Open a nested scope: emits `* Name {` and bumps indent.
#define SYNTH_OPEN_SUB(name)                                                   \
  LLVM_DEBUG(::cudaq::synth::open_scope((name), /*sub=*/true))

/// Close a scope with no outcome -- emits a bare `}`.
#define SYNTH_CLOSE() LLVM_DEBUG(::cudaq::synth::close_scope({}, {}))

/// Close a scope with a success outcome: `} -> success : reason`. `reason`
/// may be a string literal, llvm::Twine-convertible expression, std::string,
/// or anything implicitly convertible to llvm::StringRef. Pass {} (or "")
/// to omit the reason and emit `} -> success` alone.
#define SYNTH_CLOSE_SUCCESS(reason)                                            \
  LLVM_DEBUG(::cudaq::synth::close_scope("success", (reason)))

/// Close a scope with a failure outcome: `} -> failure : reason`.
#define SYNTH_CLOSE_FAILURE(reason)                                            \
  LLVM_DEBUG(::cudaq::synth::close_scope("failure", (reason)))

/// Horizontal rule between major iterations.
#define SYNTH_FENCE() LLVM_DEBUG(::cudaq::synth::fence_line())

/// One-line action marker `** Verb : details`. Returns the prefixed stream
/// already past `** Verb : `; the caller chains further `<<` for the
/// details and a trailing newline. Self-gated via the LLVM_DEBUG check so
/// it can be used outside an explicit LLVM_DEBUG block.
#define SYNTH_ACTION(verb)                                                     \
  for (bool _synth_act = ::llvm::DebugFlag &&                                  \
                         ::llvm::isCurrentDebugType("cudaq-synth");            \
       _synth_act; _synth_act = false)                                         \
  ::cudaq::synth::dbgs() << "** " << (verb) << " : "

#else // NDEBUG

#define SYNTH_OPEN(name) ((void)0)
#define SYNTH_OPEN_SUB(name) ((void)0)
#define SYNTH_CLOSE() ((void)0)
#define SYNTH_CLOSE_SUCCESS(reason) ((void)0)
#define SYNTH_CLOSE_FAILURE(reason) ((void)0)
#define SYNTH_FENCE() ((void)0)
#define SYNTH_ACTION(verb)                                                     \
  for (bool _synth_act = false; _synth_act; _synth_act = false)                \
  ::llvm::nulls()

#endif // NDEBUG
