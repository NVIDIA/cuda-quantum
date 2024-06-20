/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/qspan.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

namespace cudaq {

#if CUDAQ_USE_STD20
namespace details {
/// `qreg<N>` for N < 1 should be a compile error
template <std::size_t N>
concept ValidQregSize = N > 0;
} // namespace details
#endif

/// @brief A `qreg` is a container for qudits. This container can be dynamic or
/// compile-time-size specified. By default, the `qreg` is constructed as a
/// dynamic register (vector-like) of qubits (2-level). This can be changed via
/// the `qreg` type template parameters.
#if CUDAQ_USE_STD20
template <std::size_t N = dyn, std::size_t Levels = 2>
  requires(details::ValidQregSize<N>)
#else
template <std::size_t N = dyn, std::size_t Levels = 2,
          typename = std::enable_if_t<(N > 0)>>
#endif
class [[deprecated(
    "The qreg type is deprecated in favor of qvector (for dynamic lengths) and "
    "qarray (for constant lengths).")]] qreg {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

private:
#if CUDAQ_USE_STD20
  /// @brief If the size is dynamic, then we use vector of qudits,
  /// if not dynamic, use an array.
  std::conditional_t<N == dyn, std::vector<value_type>,
                     std::array<value_type, N>>
      qudits;
#else
  std::vector<value_type> qudits;
#endif

public:
#if CUDAQ_USE_STD20
  /// Construct a `qreg` with \p size qudits in the |0> state. Can only be used
  /// for `dyn` sized `qregs`.
  qreg(std::size_t size)
    requires(N == dyn)
      : qudits(size) {}

  /// Nullary constructor. Can only be used for `qreg`<N> q;
  qreg()
    requires(N != dyn)
  {}

  /// @cond
  /// Nullary constructor
  /// meant to be used with `kernel_builder<cudaq::qreg<>>`
  /// @endcond
  qreg()
    requires(N == dyn)
      : qudits(1) {}
#else
  qreg(std::size_t size) : qudits(size) {}

  qreg() {}
#endif

  // A `qreg` cannot be copied.
  qreg(qreg const &) = delete;
  // A `qreg` cannot be moved.
  qreg(qreg &&) = delete;

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @brief Returns the `[0, count)` qudits.
  qspan<dyn, Levels> front(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(0, count);
#else
    return {qudits.begin(), count};
#endif
  }

  /// @brief  Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits.
  qspan<dyn, Levels> back(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(size() - count, count);
#else
    return {qudits.end() - count, count};
#endif
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits.
  qspan<dyn, Levels> slice(std::size_t start, std::size_t size) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(start, size);
#else
    return {qudits.begin() + start, size};
#endif
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() { qudits.clear(); }
};

#if !CUDAQ_USE_STD20
template <>
qreg<dyn, 2>::qreg() : qudits(1) {}
#endif

// Provide the default qreg q(SIZE) deduction guide
qreg(std::size_t) -> qreg<dyn, 2>;
} // namespace cudaq

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
