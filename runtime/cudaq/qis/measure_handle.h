/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>

namespace cudaq {

namespace details {
/// Tag type used to dispatch the index-taking `measure_handle` constructor,
/// keeping `measure_handle{42}` uncompilable in user code. The tag surface is
/// reserved for internal runtime use: the library-mode `mz`/`mx`/`my` bodies
/// construct a handle from the measured bit via this constructor, and any
/// future remote-QPU adapter that reconstructs a handle from an
/// over-the-wire index uses the same surface. Inside `__qpu__` regions the
/// bridge never calls this constructor because `mz`/`mx`/`my` produce
/// `!cc.measure_handle` SSA values directly, not C++ objects.
struct handle_index_t {};
inline constexpr handle_index_t handle_index{};
} // namespace details

/// @brief Handle for a measurement event with deferred discrimination.
///
/// `measure_handle` is the return type of `cudaq::mz`, `cudaq::mx`, and
/// `cudaq::my`. It names a measurement event without committing to a
/// particular classical representation of its outcome; the classical bit is
/// produced by an implicit conversion to `bool` at whichever source-level
/// context demands it (spec `measure_handle`, Operational Semantics).
/// `measure_handle` lowers to the IR alias `!cc.measure_handle` during AST
/// -> Quake conversion and is replaced with the bare `i64` payload by
/// `lower-cc-measure-handle` immediately before QIR conversion.
///
/// A default-constructed `measure_handle` is *unbound*: it has not been
/// produced by any `mz`/`mx`/`my` call, and its `index` carries the
/// sentinel `std::numeric_limits<std::int64_t>::max()`. Reaching any
/// bool-coercion context with an unbound handle is a frontend diagnostic:
/// `discriminating an unbound measure_handle`. The tag-dispatched
/// constructor `measure_handle(details::handle_index, idx)` is reserved for
/// internal use; `measure_handle{42}` does not compile.
///
/// `operator bool()` is the sole conversion surface. It is non-explicit:
/// every `bool b = mz(q);`, `if (mz(q))`, `return mz(q);` etc. call site
/// continues to compile, and every bool-requiring context discriminates
/// the handle via this operator. No `operator==` / `operator!=` is
/// declared: equality flows through the implicit `bool` conversion, so
/// `h1 == h2` yields outcome equality (two separate `quake.discriminate`
/// ops followed by `arith.cmpi eq` in IR), not handle identity.
///
/// In MLIR-compiler mode (`nvq++`) `index` is the measurement-event index
/// the spec describes, and the AST bridge replaces every bool-coercion
/// site with `quake.discriminate` inside `__qpu__` regions -- so the
/// `operator bool()` body below is never reached from device code. At
/// host scope the only `measure_handle` values reachable under MLIR mode
/// are default-constructed (the host-device boundary rule forbids any
/// other shape), so the operator is effectively dead code there and the
/// sentinel guard returns `false`.
///
/// In every non-MLIR build (library mode with `CUDAQ_LIBRARY_MODE`
/// defined, and host-side unit tests that include this header without
/// the bridge) the inline `mz`/`mx`/`my` bodies in `qubit_qis.h` pack
/// the measured bit into the low bit of `index` (an
/// implementation-defined encoding the spec §Library Mode allows) and
/// the body below extracts that bit. Because the bridge intercepts the
/// coercion *before* this body is emitted, reading `(index & 1)` is
/// always safe: either we are in a non-MLIR build where the bit was
/// packed, or we are in MLIR mode on a default-constructed handle (where
/// the sentinel guard short-circuits to `false`).
class measure_handle {
public:
  measure_handle() = default;
  explicit measure_handle(details::handle_index_t, std::int64_t idx)
      : index(idx) {}

  operator bool() const {
    // Guard against the unbound sentinel first: `max()` is odd on
    // two's-complement `int64_t`, so without this guard an unbound
    // handle would coerce to `true`. The remaining bit extraction is
    // the library-mode / host-test encoding; in MLIR-compiler mode this
    // body is unreachable from device code and only ever sees
    // default-constructed host-scope handles, which the guard rejects.
    return index != std::numeric_limits<std::int64_t>::max() &&
           ((index & 1) != 0);
  }

private:
  // In MLIR-compiler mode `index` is the measurement-event identity
  // consumed by `!cc.measure_handle` lowering; in library mode and
  // host-side unit tests it is also the packed-bit payload read by
  // `operator bool()` above. Both readers use the same field; no
  // mode-specific diagnostic pragma is needed.
  std::int64_t index = std::numeric_limits<std::int64_t>::max();
};

} // namespace cudaq
