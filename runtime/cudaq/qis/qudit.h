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
  /// Construct a qudit. When constructing a qudit we request an allocation to
  /// the exercution manager, and receive back an unique identifier for the
  /// qudit.
  qudit() : idx(getExecutionManager()->allocateQudit(n_levels())) {}

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

  /// Destructs a qudit and request a deallocation to the execution manager.
  ~qudit() { getExecutionManager()->deallocateQudit({n_levels(), idx}); }
};

// A qubit is a qudit with 2 levels.
using qubit = qudit<2>;

} // namespace cudaq
