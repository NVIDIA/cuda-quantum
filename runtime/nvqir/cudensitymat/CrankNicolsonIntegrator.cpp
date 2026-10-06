/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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

// Crank-Nicolson predictor-corrector method.
// Reference: https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method

using cudmIntHelp = CuDensityMatIntegratorHelper;

crank_nicolson::crank_nicolson(int num_corrector_steps,
                               const std::optional<double> &max_step_size)
    : m_t(0.0), m_num_corrector_steps(num_corrector_steps),
      m_dt(max_step_size) {
  if (m_num_corrector_steps < 1)
    throw std::invalid_argument(
        "crank_nicolson integrator requires at least 1 corrector step.");
  if (m_dt.has_value() && !(*m_dt > 0.0))
    throw std::invalid_argument("max_step_size must be positive.");
}

std::shared_ptr<base_integrator> crank_nicolson::clone() {
  auto clone = std::make_shared<cudaq::integrators::crank_nicolson>();
  clone->m_num_corrector_steps = this->m_num_corrector_steps;
  clone->m_dt = this->m_dt;
  clone->m_t = this->m_t;
  // Integration updates the state in place, so the clone needs its own copy.
  if (m_state)
    cudmIntHelp::setState(clone->m_state, clone->m_t, *m_state, m_t);
  clone->m_system = this->m_system;
  clone->m_schedule = this->m_schedule;
  return clone;
}

void crank_nicolson::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
}

std::pair<double, cudaq::state> crank_nicolson::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

void crank_nicolson::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer(
      "crank_nicolson::integrate");
  cudmIntHelp::ensureStepper(m_stepper, m_state, m_system, m_schedule);
  auto &stepper = cudmIntHelp::asCudmStepper(m_stepper);

  const double startTime = m_t;
  const auto numSubSteps =
      cudmIntHelp::subStepCount(startTime, targetTime, m_dt);
  for (std::int64_t subStep = 1; subStep <= numSubSteps; ++subStep) {
    const double nextTime =
        cudmIntHelp::subStepTime(startTime, targetTime, subStep, numSubSteps);
    const double step_size = nextTime - m_t;
    auto &castSimState = *cudmIntHelp::asCudmState(*m_state);
    auto &k1 = stepper.workspaceState(0, castSimState);
    auto &k2 = stepper.workspaceState(1, castSimState);
    auto *rho_iter = &stepper.workspaceState(2, castSimState);
    auto *rho_next = &stepper.workspaceState(3, castSimState);

    auto params = cudmIntHelp::scheduleParamsAt(m_schedule, m_t);
    stepper.computeInto(castSimState, k1, m_t, params);

    auto params_next =
        cudmIntHelp::scheduleParamsAt(m_schedule, m_t + step_size);

    rho_iter->copy_from(castSimState);
    rho_iter->accumulate_inplace(k1, step_size);

    for (int iter = 0; iter < m_num_corrector_steps; ++iter) {
      stepper.computeInto(*rho_iter, k2, m_t + step_size, params_next);

      rho_next->copy_from(castSimState);
      rho_next->accumulate_inplace(k1, step_size / 2.0);
      rho_next->accumulate_inplace(k2, step_size / 2.0);

      std::swap(rho_iter, rho_next);
    }

    castSimState.swap(*rho_iter);
    m_t = nextTime;
  }
}

} // namespace integrators
} // namespace cudaq
