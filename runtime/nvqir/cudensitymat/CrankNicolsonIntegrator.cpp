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

using cudmIntHelp = CuDensityMatIntegratorHelper;

crank_nicolson::crank_nicolson(int num_corrector_steps,
                               const std::optional<double> &max_step_size)
    : m_t(0.0), m_num_corrector_steps(num_corrector_steps),
      m_dt(max_step_size) {
  if (m_num_corrector_steps < 1)
    throw std::invalid_argument(
        "crank_nicolson integrator requires at least 1 corrector step.");
}

std::shared_ptr<base_integrator> crank_nicolson::clone() {
  auto clone = std::make_shared<cudaq::integrators::crank_nicolson>();
  clone->m_num_corrector_steps = this->m_num_corrector_steps;
  clone->m_dt = this->m_dt;
  clone->m_t = this->m_t;
  clone->m_state = this->m_state;
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

  while (m_t < targetTime) {
    const double step_size =
        cudmIntHelp::computeStepSize(m_t, targetTime, m_dt);
    auto &castSimState = *cudmIntHelp::asCudmState(*m_state);

    auto params = cudmIntHelp::scheduleParamsAt(m_schedule, m_t);
    auto k1State = m_stepper->compute(*m_state, m_t, params);
    auto &k1 = *cudmIntHelp::asCudmState(k1State);

    auto params_next =
        cudmIntHelp::scheduleParamsAt(m_schedule, m_t + step_size);

    auto rho_iter_ptr = CuDensityMatState::clone(castSimState);
    rho_iter_ptr->accumulate_inplace(k1, step_size);
    auto rho_iter = std::make_shared<cudaq::state>(rho_iter_ptr.release());

    for (int iter = 0; iter < m_num_corrector_steps; ++iter) {
      auto k2State =
          m_stepper->compute(*rho_iter, m_t + step_size, params_next);
      auto &k2 = *cudmIntHelp::asCudmState(k2State);

      auto rho_next = CuDensityMatState::clone(castSimState);
      rho_next->accumulate_inplace(k1, step_size / 2.0);
      rho_next->accumulate_inplace(k2, step_size / 2.0);

      rho_iter = std::make_shared<cudaq::state>(rho_next.release());
    }

    m_state = rho_iter;
    m_t += step_size;
  }
}

} // namespace integrators
} // namespace cudaq
