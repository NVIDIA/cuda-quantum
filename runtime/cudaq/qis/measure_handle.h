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
/// reserved for internal runtime use (e.g. a future shot-level
/// result-reporting path that surfaces backend-assigned indices back to host
/// code, or a remote-QPU adapter reconstructing a handle from its
/// over-the-wire index); the bridge never calls this constructor because
/// inside kernels `mz_handle` produces a `!cc.measure_handle` SSA value
/// directly, not a C++ object.
struct handle_index_t {};
inline constexpr handle_index_t handle_index{};
} // namespace details

/// @brief Opaque, device-only handle for a measurement event.
///
/// `measure_handle` is the C++ surface for the deferred-discrimination
/// measurement API specified in the `measure_handle` Bikeshed proposal. It
/// names a measurement event without committing to a particular classical
/// representation of its outcome; the only way to obtain a classical bit from
/// a handle is to call `cudaq::discriminate(h)` (declared in
/// `cudaq/qis/qubit_qis.h`).
///
/// The `*_handle` measurement family is additive: `mz`/`mx`/`my` continue to
/// return `cudaq::measure_result` and to inline `quake.discriminate` at the
/// call site, while `mz_handle`/`mx_handle`/`my_handle` return this type and
/// emit `quake.mz` (or its `mx`/`my` siblings) only. `measure_handle` lowers
/// to the IR alias `!cc.measure_handle` (an `i64` payload in the CC dialect)
/// during AST -> Quake conversion and is replaced with the bare `i64`
/// payload by `lower-cc-measure-handle` immediately before QIR conversion.
///
/// A default-constructed `measure_handle` is *unbound*: it has not been
/// produced by any `mz_handle` / `mx_handle` / `my_handle` call, and its
/// `index` carries the sentinel `std::numeric_limits<std::int64_t>::max()`.
/// Passing an unbound handle to `cudaq::discriminate` (directly or via
/// `cudaq::to_integer` on a vector containing one) is a frontend diagnostic:
/// `discriminating an unbound measure_handle`. The tag-dispatched
/// constructor `measure_handle(details::handle_index, idx)` is reserved for
/// internal use; `measure_handle{42}` does not compile.
///
/// The handle API is MLIR-only: the operations declared here have no
/// definitions in library mode. A non-MLIR call resolves to an unresolved
/// symbol at link time, which is the intended device-only signal. The class
/// itself is trivially copyable and constructible on the host so that
/// container code (e.g. `std::vector<measure_handle>`) compiles outside of
/// `__qpu__` regions.
///
/// `operator==` and `operator!=` are deleted: comparing two handles would
/// require a classical readout, and silently inserting two `discriminate`
/// ops would defeat the explicit-discrimination invariant the type exists
/// to enforce. Users who want outcome equality write
/// `cudaq::discriminate(h1) == cudaq::discriminate(h2)` instead.
class measure_handle {
public:
  measure_handle() = default;
  explicit measure_handle(details::handle_index_t, std::int64_t idx)
      : index(idx) {}

  bool operator==(const measure_handle &) const = delete;
  bool operator!=(const measure_handle &) const = delete;

private:
  // The `index` field is only ever read from the IR (via
  // `!cc.measure_handle`), never from the host class surface. Clang fires
  // `-Wunused-private-field` on the host build (it trips
  // `test/AST-error/wall.cpp`); GCC has no equivalent diagnostic but rejects
  // `[[maybe_unused]]` on non-static data members under `-Werror=attributes`
  // in this configuration. A clang-scoped pragma keeps the diagnostic
  // suppressed for both compilers without leaning on the attribute. The
  // pragma can be dropped once a host-side accessor surfaces
  // backend-assigned indices.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
  std::int64_t index = std::numeric_limits<std::int64_t>::max();
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
};

} // namespace cudaq
