/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "qudit.h"
#include <ranges>
#include <span>

namespace cudaq {

// The qview represents a non-owning container of qudits.
template <std::size_t Levels = 2>
class qview {
public:
  // Useful typedef exposing the underlying qudit type
  // that this qview contains.
  using value_type = qudit<Levels>;

private:
  /// @brief Reference to the non-owning span of qudits
  std::span<value_type> qudits;

public:
  /// @brief Construct a qview that refers to the qudits in `other`.
  template <typename R>
    requires(std::ranges::range<R>)
  qview(R &&other) : qudits(other.begin(), other.end()) {}

  /// @brief Copy constructor
  qview(qview const &other) : qudits(other.qudits) {}

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }

  /// @brief Iterator interface, end.
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @brief Returns the `[0, count)` qudits as a new qview.
  qview<Levels> front(std::size_t count) { return qudits.first(count); }

  /// @brief Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits.
  qview<Levels> back(std::size_t count) { return qudits.last(count); }

  // Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+count)` qudits.
  qview<Levels> slice(std::size_t start, std::size_t count) {
    return qudits.subspan(start, count);
  }

  /// @brief Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }
};
} // namespace cudaq
