/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "qudit.h"
#include <ranges>
#include <span>

namespace cudaq {
inline constexpr auto dyn = std::dynamic_extent;

// The qspan represents a non-owning container of qudits. As such
// it models both dynamically allocated qudit registers as well as
// compile-time sized qudit registers.
template <std::size_t N = dyn, std::size_t Levels = 2>
class qspan {
public:
  // Useful typedef exposing the underlying qudit type
  // that this span contains.
  using value_type = qudit<Levels>;

private:
  // Reference to the non-owning span of qudits.
  std::span<value_type, N> qudits;

public:
  // Construct a span that refers to the qudits in `other`.
  template <typename R>
    requires(std::ranges::range<R>)
  qspan(R &&other) : qudits(other.begin(), other.end()) {}

  // Copy constructor
  qspan(qspan const &other) : qudits(other.qudits) {}

  // Iterator interface.
  auto begin() { return qudits.begin(); }
  auto end() { return qudits.end(); }

  // Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  // Returns the `[0, count)` qudits as a new span.
  qspan<dyn, Levels> front(std::size_t count) { return qudits.first(count); }

  // Returns the first qudit.
  value_type &front() { return qudits.front(); }

  // Returns the `[count, size())` qudits.
  qspan<dyn, Levels> back(std::size_t count) { return qudits.last(count); }
  // Returns the last qudit.
  value_type &back() { return qudits.back(); }

  // Returns the `[start, start+count)` qudits.
  qspan<dyn, Levels> slice(std::size_t start, std::size_t count) {
    return qudits.subspan(start, count);
  }

  // FIXME implement this
  // Returns the `{start, start + stride, ...}` qudits.
  //   qspan<dyn, Levels>
  //   slice(std::size_t start, std::size_t stride, std::size_t end);

  // Returns the number of contained qudits.
  std::size_t size() const { return qudits.size(); }
};
} // namespace cudaq
