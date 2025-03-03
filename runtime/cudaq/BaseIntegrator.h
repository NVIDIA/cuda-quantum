/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "operators.h"
#include "schedule.h"
#include <map>
#include <memory>
#include <vector>

namespace cudaq {
class BaseIntegrator {
public:
  /// @brief Default constructor
  BaseIntegrator() = default;

  virtual ~BaseIntegrator() = default;

  /// @brief Set the initial state and time
  virtual void setState(cudaq::state initialState, double t0) = 0;

  /// @brief Perform integration to the target time.
  virtual void integrate(double targetTime) = 0;

  /// @brief Get the current time and state.
  virtual std::pair<double, cudaq::state> getState() = 0;
};
} // namespace cudaq
