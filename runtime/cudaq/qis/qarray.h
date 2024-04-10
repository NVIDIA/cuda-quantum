/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/qview.h"

namespace cudaq {

#if CUDAQ_USE_STD20
namespace details {
/// `qarray`<N> for N < 1 should be a compile error
template <std::size_t N>
concept ValidQArraySize = N > 0;
} // namespace details
#endif

/// @brief A `qarray` is an owning, compile-time sized container for qudits.
/// The semantics of the `qarray` follows that of a `std::array` for qudits. It
/// is templated on the number of qudits contained and the number of levels for
/// the held qudits.
template <std::size_t N, std::size_t Levels = 2>
#if CUDAQ_USE_STD20
  requires(details::ValidQArraySize<N>)
#endif
class qarray {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

private:
  /// @brief Reference to the held / owned array of qudits
  std::array<value_type, N> qudits;

public:
  /// Nullary constructor
  qarray() {}

  /// @brief `qarray` cannot be copied
  qarray(const qarray &) = delete;

  /// @brief `qarray` cannot be moved
  qarray(qarray &&) = delete;

  /// @brief `qarray` cannot be copy assigned.
  qarray &operator=(const qarray &) = delete;

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }

  /// @brief Iterator interface, end.
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @return the `[0, count)` qudits as a non-owning `qview`.
  qview<Levels> front(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(0, count);
#else
    return {qudits.begin(), count};
#endif
  }

  /// @return the first qudit.
  value_type &front() { return qudits.front(); }

  /// @return the `[count, size())` qudits as a non-owning `qview`.
  qview<Levels> back(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(size() - count, count);
#else
    return {qudits.end() - count, count};
#endif
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @return the `[start, start+size)` qudits as a non-owning `qview`
  qview<Levels> slice(std::size_t start, std::size_t size) {
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

} // namespace cudaq
