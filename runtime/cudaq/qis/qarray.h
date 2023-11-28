/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/qview.h"
#include "host_config.h"

namespace cudaq {

#if CUDAQ_USE_STD20
namespace details {
/// qarray<N> for N < 1 should be a compile error
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

  /// @brief Returns the `[0, count)` qudits as a non-owning qview.
  qview<Levels> front(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(0, count);
#else
    typename std::vector<value_type>::const_iterator first = qudits.begin();
    typename std::vector<value_type>::const_iterator last =
        qudits.begin() + count;
    return {qudits(first, last)};
#endif
  }

  /// @brief Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits as a non-owning qview
  qview<Levels> back(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(size() - count, count);
#else
    typename std::vector<value_type>::const_iterator first =
        qudits.end() - count;
    typename std::vector<value_type>::const_iterator last = qudits.end();
    return {qudits(first, last)};
#endif
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits as a non-owning qview
  qview<Levels> slice(std::size_t start, std::size_t size) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(start, size);
#else
    typename std::vector<value_type>::const_iterator first =
        qudits.begin() + start;
    typename std::vector<value_type>::const_iterator last =
        qudits.begin() + start + size;
    return {qudits(first, last)};
#endif
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }

  /// @brief Destroys all contained qudits. Postcondition: `size() == 0`.
  void clear() { qudits.clear(); }
};

} // namespace cudaq
