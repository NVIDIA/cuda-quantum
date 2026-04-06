/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/runtime/logger/logger.h"

namespace cudaq {
namespace integrators {

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

void magnus_expansion::setState(const cudaq::state &initial_state, double t0) {
  auto *simState = cudaq::state_helper::getSimulationState(
      const_cast<cudaq::state *>(&initial_state));
  auto *cudmState = dynamic_cast<CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  m_state = std::make_shared<cudaq::state>(
      CuDensityMatState::clone(*cudmState).release());
  m_t = t0;
}

std::pair<double, cudaq::state> magnus_expansion::getState() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  return std::make_pair(
      m_t, cudaq::state(CuDensityMatState::clone(*castSimState).release()));
}

void magnus_expansion::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer(
      "magnus_expansion::integrate");

  const auto asCudmState = [](cudaq::state &cudaqState) -> CuDensityMatState * {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  };

  std::unordered_map<std::string, std::complex<double>> params;

  if (!m_stepper) {
    auto &castSimState = *asCudmState(*m_state);
    for (const auto &param : m_schedule.get_parameters())
      params[param] = m_schedule.get_value_function()(param, 0.0);

    auto liouvillian =
        m_system.superOp.has_value()
            ? cudaq::dynamics::Context::getCurrentContext()
                  ->getOpConverter()
                  .constructLiouvillian({m_system.superOp.value()},
                                        m_system.modeExtents, params)
            : cudaq::dynamics::Context::getCurrentContext()
                  ->getOpConverter()
                  .constructLiouvillian({m_system.hamiltonian},
                                        {m_system.collapseOps},
                                        m_system.modeExtents, params,
                                        castSimState.is_density_matrix());
    m_stepper = std::make_unique<CuDensityMatTimeStepper>(
        asCudmState(*m_state)->get_handle(), liouvillian);
  }

  while (m_t < targetTime) {
    const double step_size =
        std::min(m_dt.value_or(targetTime - m_t), targetTime - m_t);

    auto &castSimState = *asCudmState(*m_state);

    std::unordered_map<std::string, std::complex<double>> params_mid;
    for (const auto &param : m_schedule.get_parameters())
      params_mid[param] =
          m_schedule.get_value_function()(param, m_t + step_size / 2.0);
    const double t_mid = m_t + step_size / 2.0;

    auto result = CuDensityMatState::clone(castSimState);

    cudaq::state v(CuDensityMatState::clone(castSimState).release());

    for (int k = 1; k <= m_num_taylor_terms; ++k) {
      auto Lv = m_stepper->compute(v, t_mid, params_mid);
      auto &Lv_cudm = *asCudmState(Lv);

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
