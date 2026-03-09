/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/base_integrator.h"
#include "cudaq/algorithms/base_time_stepper.h"
#include "cudaq/operators.h"
#include <memory>

namespace cudaq {
namespace integrators {

class runge_kutta : public cudaq::base_integrator {
public:
  /// @brief The default `Runge-Kutta` integration order.
  // Note: we use 4th order as the default since (1) it produces better
  // convergence/stability and (2) that is the typical order that the name
  // `Runge-Kutta` is associated with (e.g., RK method can be generalized to
  // cover 1st order Euler method)
  static constexpr int default_order = 4;
  /// @brief Constructor
  // (1) Integration order
  // (2) Max step size: if none provided, the schedule of time points where we
  // want to compute and save intermediate results will be used. If provided,
  // the integrator will make sub-steps no larger than this value to integrate
  // toward scheduled time points.
  runge_kutta(int order = default_order,
              const std::optional<double> &max_step_size = {});
  /// @brief Integrate toward a specified time point.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current state of the integrator
  // Returns the current time point and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator.
  std::shared_ptr<base_integrator> clone() override;

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  int m_order;
  std::optional<double> m_dt;
};
} // namespace integrators
} // namespace cudaq
