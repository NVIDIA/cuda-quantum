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

using H = CuDensityMatIntegratorHelper;

runge_kutta::runge_kutta(int order, const std::optional<double> &max_step_size)
    : m_t(0.0), m_order(order), m_dt(max_step_size) {
  if (m_order != 1 && m_order != 2 && m_order != 4)
    throw std::invalid_argument(
        "runge_kutta integrator only supports integration order 1, 2, or 4.");
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
  H::setState(m_state, m_t, initialState, t0);
}

std::pair<double, cudaq::state> runge_kutta::getState() {
  return H::getState(m_state, m_t);
}

void runge_kutta::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("runge_kutta::integrate");
  H::ensureStepper(m_stepper, m_state, m_system, m_schedule);
  auto &castSimState = *H::asCudmState(*m_state);

  while (m_t < targetTime) {
    const double step_size = H::computeStepSize(m_t, targetTime, m_dt);
    if (m_order == 1) {
      // Euler method (1st order)
      auto params = H::scheduleParamsAt(m_schedule, m_t);
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *H::asCudmState(k1State);
      k1 *= step_size;
      castSimState += k1;
    } else if (m_order == 2) {
      // Midpoint method (2nd order)
      // Standard formula: y_{n+1} = y_n + h * k2
      // where k1 = f(t, y_n), k2 = f(t + h/2, y_n + h/2 * k1)
      auto params = H::scheduleParamsAt(m_schedule, m_t);
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *H::asCudmState(k1State);

      // Create temporary state: y_temp = y_n + (h/2) * k1
      auto rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp->accumulate_inplace(k1, step_size / 2.0);

      // Compute k2 at the midpoint
      auto params_mid = H::scheduleParamsAt(m_schedule, m_t + step_size / 2.0);
      auto k2State = m_stepper->compute(cudaq::state(rho_temp.release()),
                                        m_t + step_size / 2.0, params_mid);
      auto &k2 = *H::asCudmState(k2State);

      // Final update: y_{n+1} = y_n + h * k2
      castSimState.accumulate_inplace(k2, step_size);
    } else if (m_order == 4) {
      // Runge-Kutta method (4th order)
      auto params = H::scheduleParamsAt(m_schedule, m_t);
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *H::asCudmState(k1State);
      auto rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp->accumulate_inplace(k1, step_size / 2); // y + h * k1/2
      auto params_mid = H::scheduleParamsAt(m_schedule, m_t + step_size / 2.0);
      auto k2State = m_stepper->compute(cudaq::state(rho_temp.release()),
                                        m_t + step_size / 2.0, params_mid);
      auto &k2 = *H::asCudmState(k2State);
      auto rho_temp_2 = CuDensityMatState::clone(castSimState);
      rho_temp_2->accumulate_inplace(k2, step_size / 2); // y + h * k2/2
      auto k3State = m_stepper->compute(cudaq::state(rho_temp_2.release()),
                                        m_t + step_size / 2.0, params_mid);
      auto &k3 = *H::asCudmState(k3State);
      auto rho_temp_3 = CuDensityMatState::clone(castSimState);
      rho_temp_3->accumulate_inplace(k3, step_size); // y + h * k3
      auto params_end = H::scheduleParamsAt(m_schedule, m_t + step_size);
      auto k4State = m_stepper->compute(cudaq::state(rho_temp_3.release()),
                                        m_t + step_size, params_end);
      auto &k4 = *H::asCudmState(k4State);

      castSimState.accumulate_inplace(k1, step_size / 6.0);
      castSimState.accumulate_inplace(k2, step_size / 3.0);
      castSimState.accumulate_inplace(k3, step_size / 3.0);
      castSimState.accumulate_inplace(k4, step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    m_t += step_size;
  }
}
} // namespace integrators
} // namespace cudaq
