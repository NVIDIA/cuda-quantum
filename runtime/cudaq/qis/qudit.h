/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "execution_manager.h"

using namespace std::complex_literals;

namespace cudaq {
using complex = std::complex<double>;

namespace ket {
inline static const std::vector<complex> zero{1. + 0i, 0. + 0i};
inline static const std::vector<complex> one{0. + 0i, 1. + 0i};
} // namespace ket

/// The qudit models a general d-level quantum system.
/// This type is templated on the number of levels d.
template <std::size_t Levels>
class qudit {
  /// Every qudit has a logical index in the global qudit register,
  /// `idx` is this logical index, it must be
  /// provided at construction and is immutable.
  const std::size_t idx = 0;

  // Bool to indicate if we are currently negated
  // as a control qudit.
  bool isNegativeControl = false;

public:
  /// Construct a qudit, will allocated a new unique index
  qudit() : idx(getExecutionManager()->allocateQudit(n_levels())) {}
  qudit(const std::vector<simulation_scalar> &state) : qudit() {
    if (state.size() != Levels)
      throw std::runtime_error(
          "Invalid number of state vector elements for qudit allocation (" +
          std::to_string(state.size()) + ").");

    auto norm =
        std::inner_product(
            state.begin(), state.end(), state.begin(),
            simulation_scalar{0., 0.}, [](auto a, auto b) { return a + b; },
            [](auto a, auto b) { return std::conj(a) * b; })
            .real();
    if (std::fabs(1.0 - norm) > 1e-4)
      throw std::runtime_error("Invalid vector norm for qudit allocation.");

    // Perform the initialization
    auto precision = std::is_same_v<simulation_scalar::value_type, float>
                         ? simulation_precision::fp32
                         : simulation_precision::fp64;
    getExecutionManager()->initializeState({QuditInfo(n_levels(), idx)},
                                           state.data(), precision);
  }
  qudit(const std::initializer_list<complex> &list)
      : qudit({list.begin(), list.end()}) {}

  // Qudits cannot be copied
  qudit(const qudit &q) = delete;
  // qudits cannot be moved
  qudit(qudit &&) = delete;

  // Return the unique id / index for this qudit
  std::size_t id() const { return idx; }

  // Return this qudit's dimension
  static constexpr std::size_t n_levels() { return Levels; }

  // Qudits used as controls can be negated, i.e
  // instead of applying a target op if the qudit is
  // in the vacuum state, you can apply a target op if the
  // state is in an excited state if this control qudit is negated
  qudit<Levels> &negate() {
    isNegativeControl = !isNegativeControl;
    return *this;
  }

  // Is this qudit negated?
  bool is_negative() { return isNegativeControl; }

  // Syntactic sugar for negating a control
  qudit<Levels> &operator!() { return negate(); }

  // Destructor, return the qudit so it can be reused
  ~qudit() { getExecutionManager()->returnQudit({n_levels(), idx}); }
};

// A qubit is a qudit with 2 levels.
using qubit = qudit<2>;

} // namespace cudaq
