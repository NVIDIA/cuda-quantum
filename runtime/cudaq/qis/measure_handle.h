/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

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
/// The handle API is MLIR-only: the operations declared here have no
/// definitions in library mode. A non-MLIR call resolves to an unresolved
/// symbol at link time, which is the intended device-only signal. The class
/// itself is constructible on the host so that container code (e.g.
/// `std::vector<measure_handle>`) compiles outside of `__qpu__` regions.
///
/// `operator==` and `operator!=` are deleted: comparing two handles would
/// require a classical readout, and silently inserting two `discriminate`
/// ops would defeat the explicit-discrimination invariant the type exists
/// to enforce. Users who want outcome equality write
/// `cudaq::discriminate(h1) == cudaq::discriminate(h2)` instead.
class measure_handle {
public:
  measure_handle() = default;
  explicit measure_handle(int idx) : index(idx) {}

  bool operator==(const measure_handle &) const = delete;
  bool operator!=(const measure_handle &) const = delete;

private:
  int index = 0;
};

} // namespace cudaq
