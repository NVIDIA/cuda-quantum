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

// Magnus expansion integrator (first-order / midpoint approximation).
// Reference: https://en.wikipedia.org/wiki/Magnus_expansion

using cudmIntHelp = CuDensityMatIntegratorHelper;

magnus_expansion::magnus_expansion(int num_taylor_terms,
                                   const std::optional<double> &max_step_size)
    : m_t(0.0), m_num_taylor_terms(num_taylor_terms), m_dt(max_step_size) {
  if (m_num_taylor_terms < 1)
    throw std::invalid_argument(
        "magnus_expansion integrator requires at least 1 Taylor term.");
}

std::shared_ptr<base_integrator> magnus_expansion::clone() {
  auto clone = std::make_shared<cudaq::integrators::magnus_expansion>();
  clone->m_num_taylor_terms = this->m_num_taylor_terms;
  clone->m_dt = this->m_dt;
  clone->m_t = this->m_t;
  clone->m_state = this->m_state;
  clone->m_system = this->m_system;
  clone->m_schedule = this->m_schedule;
  return clone;
}

void magnus_expansion::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
}

std::pair<double, cudaq::state> magnus_expansion::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

void magnus_expansion::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer(
      "magnus_expansion::integrate");
  cudmIntHelp::ensureStepper(m_stepper, m_state, m_system, m_schedule);

  while (m_t < targetTime) {
    const double step_size =
        cudmIntHelp::computeStepSize(m_t, targetTime, m_dt);
    auto &castSimState = *cudmIntHelp::asCudmState(*m_state);

    const double t_mid = m_t + step_size / 2.0;
    auto params_mid = cudmIntHelp::scheduleParamsAt(m_schedule, t_mid);

    auto result = CuDensityMatState::clone(castSimState);
    cudaq::state v(CuDensityMatState::clone(castSimState).release());

    for (int k = 1; k <= m_num_taylor_terms; ++k) {
      auto Lv = m_stepper->compute(v, t_mid, params_mid);
      auto &Lv_cudm = *cudmIntHelp::asCudmState(Lv);

      Lv_cudm *= (step_size / static_cast<double>(k));
      result->accumulate_inplace(Lv_cudm, 1.0);

      v = std::move(Lv);
    }

    m_state = std::make_shared<cudaq::state>(result.release());
    m_t += step_size;
  }
}

} // namespace integrators
} // namespace cudaq
