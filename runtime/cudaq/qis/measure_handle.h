/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// `cudaq::measure_handle` is the MLIR-mode only handle, library-mode is not
// supported.
#ifndef CUDAQ_LIBRARY_MODE

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>

namespace cudaq::details {
/// Tag type used to dispatch the index-taking `measure_handle` constructor,
/// so `measure_handle{42}` cannot be compiled in user code. The tag surface is
/// reserved for internal runtime use. Inside `__qpu__` regions the
/// bridge never calls this constructor because `mz`/`mx`/`my` produce
/// `!cc.measure_handle` SSA values directly.
struct handle_index_t {};
inline constexpr handle_index_t handle_index{};
} // namespace cudaq::details

namespace cudaq {

/// @brief Handle for a measurement event with deferred discrimination.
///
/// `measure_handle` is the return type of `mz`, `my`, and `mx`.
/// `measure_handle` lowers to the IR alias `!cc.measure_handle` during AST
/// -> Quake conversion and is converted to the bare `i64` payload by the
/// QIR API conversion's type converter.
///
/// A default-constructed `measure_handle` is *unbound*: it has not been
/// produced by any `mz`/`mx`/`my` call, and its `index` carries the
/// sentinel `std::numeric_limits<std::int64_t>::max()`.
///
class measure_handle {
public:
  measure_handle() noexcept : index(std::numeric_limits<std::int64_t>::max()) {}

  explicit measure_handle(details::handle_index_t, std::int64_t idx)
      : index(idx) {}

  // The bridge intercepts every `bool` coercion of a `measure_handle` and
  // emits `quake.discriminate`. This body therefore must never run in a
  // built kernel; reaching it means the bridge missed an interception
  // path and the program would otherwise compute on a meaningless `bool`.
  // Abort instead of returning a quiet wrong answer.
  operator bool() const { std::abort(); }

private:
  //`index` is the measurement-event identity consumed by `!cc.measure_handle`
  // lowering, the encoding is implementation-defined; the bridge produces the
  // SSA value directly and never reads this field from C++.
#if defined(__clang__)
  [[maybe_unused]]
#endif
  std::int64_t index = std::numeric_limits<std::int64_t>::max();
};

// Compile-time checks: the bridge marshals `measure_handle` as a bare `i64`.
static_assert(
    sizeof(measure_handle) == sizeof(std::int64_t),
    "`cudaq::measure_handle` must have the `i64` payload width assumed "
    "by the IR-mode lowering");
static_assert(std::is_trivially_copyable_v<measure_handle>,
              "`cudaq::measure_handle` must be trivially copyable so the "
              "bridge can marshal it as a contiguous `i64` payload");
static_assert(
    std::is_standard_layout_v<measure_handle>,
    "`cudaq::measure_handle` must be standard layout so the host-side "
    "ABI offset of any aggregate member matches the IR lowering");

} // namespace cudaq

#endif // !CUDAQ_LIBRARY_MODE
