/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatIntegratorBase.h"
#include "CuDensityMatUtils.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/runtime/logger/logger.h"

namespace cudaq {
namespace integrators {

using cudmIntHelp = CuDensityMatIntegratorHelper;

runge_kutta::runge_kutta(int order, const std::optional<double> &max_step_size)
    : m_t(0.0), m_order(order), m_dt(max_step_size) {
  if (m_order != 1 && m_order != 2 && m_order != 4)
    throw std::invalid_argument(
        "runge_kutta integrator only supports integration order 1, 2, or 4.");
  if (m_dt.has_value() && !(*m_dt > 0.0))
    throw std::invalid_argument("max_step_size must be positive.");
}

std::shared_ptr<base_integrator> runge_kutta::clone() {
  auto clone = std::make_shared<cudaq::integrators::runge_kutta>();
  clone->m_order = this->m_order;
  clone->m_dt = this->m_dt;
  clone->m_t = this->m_t;
  clone->m_state = this->m_state;
  clone->m_system = this->m_system;
  clone->m_schedule = this->m_schedule;
  return clone;
}

void runge_kutta::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
}

std::pair<double, cudaq::state> runge_kutta::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

void runge_kutta::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("runge_kutta::integrate");
  cudmIntHelp::ensureStepper(m_stepper, m_state, m_system, m_schedule);
  auto &stepper = cudmIntHelp::asCudmStepper(m_stepper);
  auto &castSimState = *cudmIntHelp::asCudmState(*m_state);

  const double startTime = m_t;
  const auto numSubSteps =
      cudmIntHelp::subStepCount(startTime, targetTime, m_dt);
  for (std::int64_t subStep = 1; subStep <= numSubSteps; ++subStep) {
    const double nextTime =
        cudmIntHelp::subStepTime(startTime, targetTime, subStep, numSubSteps);
    const double step_size = nextTime - m_t;
    if (m_order == 1) {
      // Euler method (1st order)
      auto params = cudmIntHelp::scheduleParamsAt(m_schedule, m_t);
      auto &k1 = stepper.workspaceState(0, castSimState);
      stepper.computeInto(castSimState, k1, m_t, params);
      castSimState.accumulate_inplace(k1, step_size);
    } else if (m_order == 2) {
      // Midpoint method (2nd order)
      // Standard formula: y_{n+1} = y_n + h * k2
      // where k1 = f(t, y_n), k2 = f(t + h/2, y_n + h/2 * k1)
      auto &k1 = stepper.workspaceState(0, castSimState);
      auto &k2 = stepper.workspaceState(1, castSimState);
      auto &rho_temp = stepper.workspaceState(2, castSimState);
      auto params = cudmIntHelp::scheduleParamsAt(m_schedule, m_t);
      stepper.computeInto(castSimState, k1, m_t, params);

      // Temporary state: y_temp = y_n + (h/2) * k1
      rho_temp.copy_from(castSimState);
      rho_temp.accumulate_inplace(k1, step_size / 2.0);

      // Compute k2 at the midpoint
      auto params_mid =
          cudmIntHelp::scheduleParamsAt(m_schedule, m_t + step_size / 2.0);
      stepper.computeInto(rho_temp, k2, m_t + step_size / 2.0, params_mid);

      // Final update: y_{n+1} = y_n + h * k2
      castSimState.accumulate_inplace(k2, step_size);
    } else if (m_order == 4) {
      // Runge-Kutta method (4th order)
      auto &k1 = stepper.workspaceState(0, castSimState);
      auto &k2 = stepper.workspaceState(1, castSimState);
      auto &k3 = stepper.workspaceState(2, castSimState);
      auto &k4 = stepper.workspaceState(3, castSimState);
      auto &rho_temp = stepper.workspaceState(4, castSimState);
      auto params = cudmIntHelp::scheduleParamsAt(m_schedule, m_t);
      stepper.computeInto(castSimState, k1, m_t, params);
      rho_temp.copy_from(castSimState);
      rho_temp.accumulate_inplace(k1, step_size / 2); // y + h * k1/2
      auto params_mid =
          cudmIntHelp::scheduleParamsAt(m_schedule, m_t + step_size / 2.0);
      stepper.computeInto(rho_temp, k2, m_t + step_size / 2.0, params_mid);
      rho_temp.copy_from(castSimState);
      rho_temp.accumulate_inplace(k2, step_size / 2); // y + h * k2/2
      stepper.computeInto(rho_temp, k3, m_t + step_size / 2.0, params_mid);
      rho_temp.copy_from(castSimState);
      rho_temp.accumulate_inplace(k3, step_size); // y + h * k3
      auto params_end =
          cudmIntHelp::scheduleParamsAt(m_schedule, m_t + step_size);
      stepper.computeInto(rho_temp, k4, m_t + step_size, params_end);

      castSimState.accumulate_inplace(k1, step_size / 6.0);
      castSimState.accumulate_inplace(k2, step_size / 3.0);
      castSimState.accumulate_inplace(k3, step_size / 3.0);
      castSimState.accumulate_inplace(k4, step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    m_t = nextTime;
  }
}
} // namespace integrators
} // namespace cudaq
