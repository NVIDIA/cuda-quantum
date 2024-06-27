/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include "cudaq/qis/qudit.h"
#if CUDAQ_USE_STD20
#include <ranges>
#include <span>
#endif

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
inline constexpr auto dyn = std::dynamic_extent;
#else
inline constexpr std::size_t dyn = ~0;
#endif

// The qspan represents a non-owning container of qudits. As such
// it models both dynamically allocated qudit registers as well as
// compile-time sized qudit registers.
template <std::size_t N = dyn, std::size_t Levels = 2>
class [[deprecated("The qspan type is deprecated in favor of qview.")]] qspan {
public:
  // Useful typedef exposing the underlying qudit type
  // that this span contains.
  using value_type = qudit<Levels>;

#if CUDAQ_USE_STD20
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

#else
  // C++11 reimplementation of a std::span.

private:
  value_type *qudits = nullptr;
  std::size_t qusize = 0;

public:
  qspan(value_type *otherQudits, std::size_t otherQusize)
      : qudits(otherQudits), qusize(otherQusize) {}
  template <typename Iterator>
  qspan(Iterator &&otherQudits, std::size_t otherQusize)
      : qudits(&*otherQudits), qusize(otherQusize) {}
  template <typename R>
  qspan(R &&other)
      : qudits(&*other.begin()),
        qusize(std::distance(other.begin(), other.end())) {}
  qspan(qspan const &other) : qudits(other.qudits), qusize(other.qusize) {}

  value_type *begin() { return qudits; }
  value_type *end() { return qudits + qusize; }
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }
  qspan<dyn, Levels> front(std::size_t count) { return {qudits, count}; }
  value_type &front() { return *qudits; }
  qspan<dyn, Levels> back(std::size_t count) {
    return {qudits + qusize - count, count};
  }
  value_type &back() { return *(qudits + qusize - 1); }
  qspan<dyn, Levels> slice(std::size_t start, std::size_t count) {
    return {qudits + start, count};
  }
  std::size_t size() const { return qusize; }
#endif
};
} // namespace cudaq

#if defined(__clang__)
#pragma clang diagnostic pop
#endif
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
