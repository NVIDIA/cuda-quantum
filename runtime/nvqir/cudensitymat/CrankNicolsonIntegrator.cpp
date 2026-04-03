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

void crank_nicolson::setState(const cudaq::state &initial_state, double t0) {
  auto *simState = cudaq::state_helper::getSimulationState(
      const_cast<cudaq::state *>(&initial_state));
  auto *cudmState = dynamic_cast<CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  m_state = std::make_shared<cudaq::state>(
      CuDensityMatState::clone(*cudmState).release());
  m_t = t0;
}

std::pair<double, cudaq::state> crank_nicolson::getState() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  return std::make_pair(
      m_t, cudaq::state(CuDensityMatState::clone(*castSimState).release()));
}

void crank_nicolson::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer(
      "crank_nicolson::integrate");

  const auto asCudmState =
      [](cudaq::state &cudaqState) -> CuDensityMatState * {
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

    for (const auto &param : m_schedule.get_parameters())
      params[param] = m_schedule.get_value_function()(param, m_t);
    auto k1State = m_stepper->compute(*m_state, m_t, params);
    auto &k1 = *asCudmState(k1State);

    std::unordered_map<std::string, std::complex<double>> params_next;
    for (const auto &param : m_schedule.get_parameters())
      params_next[param] =
          m_schedule.get_value_function()(param, m_t + step_size);

    auto rho_iter_ptr = CuDensityMatState::clone(castSimState);
    rho_iter_ptr->accumulate_inplace(k1, step_size);
    auto rho_iter =
        std::make_shared<cudaq::state>(rho_iter_ptr.release());

    for (int iter = 0; iter < m_num_corrector_steps; ++iter) {
      auto k2State =
          m_stepper->compute(*rho_iter, m_t + step_size, params_next);
      auto &k2 = *asCudmState(k2State);

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