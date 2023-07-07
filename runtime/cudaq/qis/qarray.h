/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "qview.h"

namespace cudaq {

namespace details {
/// qarray<N> for N < 1 should be a compile error
template <std::size_t N>
concept ValidQArraySize = N > 0;
} // namespace details

/// @brief A `qarray` is an owning, compile-time sized container for qudits.
/// The semantics of the `qarray` follows that of a `std::array` for qudits. It
/// is templated on the number of qudits contained and the number of levels for
/// the held qudits.
template <std::size_t N, std::size_t Levels = 2>
  requires(details::ValidQArraySize<N>)
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

  /// @brief Returns the `[0, count)` qudits as a non-owning qview.
  qview<Levels> front(std::size_t count) {
    return std::span(qudits).subspan(0, count);
  }

  /// @brief Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits as a non-owning qview
  qview<Levels> back(std::size_t count) {
    return std::span(qudits).subspan(size() - count, count);
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits as a non-owning qview
  qview<Levels> slice(std::size_t start, std::size_t size) {
    return std::span(qudits).subspan(start, size);
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() { qudits.clear(); }
};

} // namespace cudaq
