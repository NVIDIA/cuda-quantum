/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include "common/Logger.h"
#include "cudaq/algorithms/integrator.h"
namespace cudaq {
namespace integrators {

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

void runge_kutta::setState(const cudaq::state &initial_state, double t0) {
  auto *simState = cudaq::state_helper::getSimulationState(
      const_cast<cudaq::state *>(&initial_state));
  auto *cudmState = dynamic_cast<CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  m_state = std::make_shared<cudaq::state>(
      CuDensityMatState::clone(*cudmState).release());
  m_t = t0;
}

std::pair<double, cudaq::state> runge_kutta::getState() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");

  return std::make_pair(
      m_t, cudaq::state(CuDensityMatState::clone(*castSimState).release()));
}

void runge_kutta::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("runge_kutta::integrate");
  const auto asCudmState = [](cudaq::state &cudaqState) -> CuDensityMatState * {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  };
  auto &castSimState = *asCudmState(*m_state);
  std::unordered_map<std::string, std::complex<double>> params;
  if (!m_stepper) {
    for (const auto &param : m_schedule.get_parameters()) {
      params[param] = m_schedule.get_value_function()(param, 0.0);
    }

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
        castSimState.get_handle(), liouvillian);
  }
  while (m_t < targetTime) {
    const double step_size =
        std::min(m_dt.value_or(targetTime - m_t), targetTime - m_t);
    if (m_order == 1) {
      // Euler method (1st order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *asCudmState(k1State);
      k1 *= step_size;
      castSimState += k1;
    } else if (m_order == 2) {
      // Midpoint method (2nd order)
      // Standard formula: y_{n+1} = y_n + h * k2
      // where k1 = f(t, y_n), k2 = f(t + h/2, y_n + h/2 * k1)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *asCudmState(k1State);

      // Create temporary state: y_temp = y_n + (h/2) * k1
      auto rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp->accumulate_inplace(k1, step_size / 2.0);

      // Compute k2 at the midpoint
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] =
            m_schedule.get_value_function()(param, m_t + step_size / 2.0);
      }
      auto k2State = m_stepper->compute(cudaq::state(rho_temp.release()),
                                        m_t + step_size / 2.0, params);
      auto &k2 = *asCudmState(k2State);

      // Final update: y_{n+1} = y_n + h * k2
      castSimState.accumulate_inplace(k2, step_size);
    } else if (m_order == 4) {
      // Runge-Kutta method (4th order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, params);
      auto &k1 = *asCudmState(k1State);
      auto rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp->accumulate_inplace(k1, step_size / 2); // y + h * k1/2
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] =
            m_schedule.get_value_function()(param, m_t + step_size / 2.0);
      }
      auto k2State = m_stepper->compute(cudaq::state(rho_temp.release()),
                                        m_t + step_size / 2.0, params);
      auto &k2 = *asCudmState(k2State);
      auto rho_temp_2 = CuDensityMatState::clone(castSimState);
      rho_temp_2->accumulate_inplace(k2, step_size / 2); // y + h * k2/2
      auto k3State = m_stepper->compute(cudaq::state(rho_temp_2.release()),
                                        m_t + step_size / 2.0, params);
      auto &k3 = *asCudmState(k3State);
      auto rho_temp_3 = CuDensityMatState::clone(castSimState);
      rho_temp_3->accumulate_inplace(k3, step_size); // y + h * k3
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t + step_size);
      }
      auto k4State = m_stepper->compute(cudaq::state(rho_temp_3.release()),
                                        m_t + step_size, params);
      auto &k4 = *asCudmState(k4State);

      castSimState.accumulate_inplace(k1, step_size / 6.0);
      castSimState.accumulate_inplace(k2, step_size / 3.0);
      castSimState.accumulate_inplace(k3, step_size / 3.0);
      castSimState.accumulate_inplace(k4, step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    // Update time
    m_t += step_size;
  }
}
} // namespace integrators
} // namespace cudaq
