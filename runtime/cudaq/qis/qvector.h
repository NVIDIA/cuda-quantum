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

class state;

/// @brief A `qvector` is an owning, dynamically sized container for qudits.
/// The semantics of the `qvector` follows that of a `std::vector` for qudits.
/// It is templated on the number of levels for the held qudits.
template <std::size_t Levels = 2>
class qvector {
public:
  /// @brief Useful typedef indicating the underlying qudit type
  using value_type = qudit<Levels>;

private:
  /// @brief Reference to the held / owned vector of qudits.
  std::vector<value_type> qudits;

public:
  /// @brief Construct a `qvector` with `size` qudits in the |0> state.
  qvector(std::size_t size) : qudits(size) {}

  qvector(const std::vector<complex> &vector)
      : qudits(std::log2(vector.size())) {
    if (Levels == 2) {
      auto numElements = std::log2(vector.size());
      if (std::floor(numElements) != numElements)
        throw std::runtime_error(
            "Invalid state vector passed to qvector initialization - number of "
            "elements must be power of 2.");
    }

    auto norm = std::inner_product(
                    vector.begin(), vector.end(), vector.begin(),
                    complex{0., 0.}, [](auto a, auto b) { return a + b; },
                    [](auto a, auto b) { return std::conj(a) * b; })
                    .real();
    if (std::fabs(1.0 - norm) > 1e-4)
      throw std::runtime_error("Invalid vector norm for qudit allocation.");

    std::vector<QuditInfo> targets;
    for (auto &q : qudits)
      targets.emplace_back(QuditInfo{Levels, q.id()});

    auto precision = std::is_same_v<complex::value_type, float>
                         ? simulation_precision::fp32
                         : simulation_precision::fp64;
    getExecutionManager()->initializeState(targets, vector.data(), precision);
  }

  qvector(const std::vector<double> &vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(std::vector<double> &&vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(const std::initializer_list<double> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}
  qvector(const std::vector<float> &vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(std::vector<float> &&vector)
      : qvector(std::vector<complex>{vector.begin(), vector.end()}) {}
  qvector(const std::initializer_list<float> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}
  qvector(const std::initializer_list<complex> &list)
      : qvector(std::vector<complex>{list.begin(), list.end()}) {}

  /// @cond
  /// Nullary constructor
  /// meant to be used with `kernel_builder<cudaq::qvector<>>`
  qvector() : qudits(1) {}
  /// @endcond

  explicit qvector(const state *);
  explicit qvector(const state &);

  /// @brief `qvectors` cannot be copied
  qvector(qvector const &) = delete;

  /// @brief `qvectors` cannot be moved
  qvector(qvector &&) = delete;

  /// @brief `qvectors` cannot be copy assigned.
  qvector &operator=(const qvector &) = delete;

  /// @brief Iterator interface, begin.
  auto begin() { return qudits.begin(); }

  /// @brief Iterator interface, end.
  auto end() { return qudits.end(); }

  /// @brief Returns the qudit at `idx`.
  value_type &operator[](const std::size_t idx) { return qudits[idx]; }

  /// @brief Returns the `[0, count)` qudits as a non-owning `qview`.
  qview<Levels> front(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(0, count);
#else
    return {qudits.begin(), count};
#endif
  }

  /// @brief Returns the first qudit.
  value_type &front() { return qudits.front(); }

  /// @brief Returns the `[count, size())` qudits as a non-owning `qview`
  qview<Levels> back(std::size_t count) {
#if CUDAQ_USE_STD20
    return std::span(qudits).subspan(size() - count, count);
#else
    return {qudits.end() - count, count};
#endif
  }

  /// @brief Returns the last qudit.
  value_type &back() { return qudits.back(); }

  /// @brief Returns the `[start, start+size)` qudits as a non-owning `qview`
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
