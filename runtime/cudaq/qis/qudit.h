/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "execution_manager.h"

namespace cudaq {

/// A handle to a qudit in the underlying quantum system.
///
/// `Qudit` objects do not carry quantum state. Instead, they encapsulate a
/// logical index that identifies a qubit in the underlying quantum system.
/// The `ExecutionManager` provides this index, which acts as a qudit unique
/// identifier through the objects' lifetimes in a kernel execution.
///
/// Note: Users of this class should _not_ write code that rely on a qudit
/// having a particular identifier. Also, the uniqueness of the identifier is
/// only guaranteed within the execution of the kernel and _not_ across
/// multiple executions of the same kernel. For example:
///
/// ```cpp
/// __qpu__ void bar() {
///   cudaq::qudit x;
/// }
///
/// __qpu__ void foo() {
///   bar();           // Here qudit `x` (in bar) could have index 0.
///   cudaq::qudit y;
///   bar();           // Here qudit `x` (in bar) could have index 1.
/// }
/// ```
template <std::size_t Levels>
class qudit {
  const std::size_t idx = 0;

  // Indicates whether this qudit is currently used as a negated control.
  bool isNegativeControl = false;

public:
  /// Construct a qudit, will allocated a new unique index
  qudit() : idx(getExecutionManager()->allocateQudit(n_levels())) {}
  qudit(const std::vector<complex> &state) : qudit() {
    if (state.size() != Levels)
      throw std::runtime_error(
          "Invalid number of state vector elements for qudit allocation (" +
          std::to_string(state.size()) + ").");

    auto norm = std::inner_product(
                    state.begin(), state.end(), state.begin(), complex{0., 0.},
                    [](auto a, auto b) { return a + b; },
                    [](auto a, auto b) { return std::conj(a) * b; })
                    .real();
    if (std::fabs(1.0 - norm) > 1e-4)
      throw std::runtime_error("Invalid vector norm for qudit allocation.");

    // Perform the initialization
    auto precision = std::is_same_v<complex::value_type, float>
                         ? simulation_precision::fp32
                         : simulation_precision::fp64;
    getExecutionManager()->initializeState({QuditInfo(n_levels(), idx)},
                                           state.data(), precision);
  }
  qudit(const std::initializer_list<complex> &list)
      : qudit({list.begin(), list.end()}) {}

  // Qudits cannot be copied
  qudit(const qudit &q) = delete;
  // Qudits cannot be moved
  qudit(qudit &&) = delete;

  /// Returns the id for this qudit.
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
