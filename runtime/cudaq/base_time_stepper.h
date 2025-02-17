/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/qis/state.h"

namespace cudaq {
template <typename TState>
class BaseTimeStepper {
public:
  virtual ~BaseTimeStepper() = default;

  /// @brief Compute the next time step for the given quantum state.
  /// @param state The quantum state to evolve.
  /// @param t Current time.
  /// @param step_size Time step size.
  /// @return The updated quantum state after stepping.
  virtual TState compute(TState &state, double t, double step_size) = 0;
};

class TimeStepper {
public:
  virtual ~TimeStepper() = default;

  virtual state
  compute(const state &inputState, double t, double step_size,
          const std::unordered_map<std::string, std::complex<double>>
              &parameters) = 0;
};
} // namespace cudaq