/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "common/Logger.h"
#include "cudaq/dynamics_integrators.h"
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
  m_state = std::make_shared<cudaq::state>(initial_state);
  m_t = t0;
}

std::pair<double, cudaq::state> runge_kutta::getState() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");

  auto cudmState =
      new CuDensityMatState(castSimState->get_handle(), *castSimState,
                            castSimState->get_hilbert_space_dims());

  return std::make_pair(m_t, cudaq::state(cudmState));
}

void runge_kutta::integrate(double targetTime) {
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
        cudaq::dynamics::Context::getCurrentContext()
            ->getOpConverter()
            .constructLiouvillian(m_system.hamiltonian, m_system.collapseOps,
                                  m_system.modeExtents, params,
                                  castSimState.is_density_matrix());
    m_stepper = std::make_unique<CuDensityMatTimeStepper>(
        castSimState.get_handle(), liouvillian);
  }
  while (m_t < targetTime) {
    const double step_size =
        std::min(m_dt.value_or(targetTime - m_t), targetTime - m_t);

    cudaq::debug("Runge-Kutta step at time {} with step size {}", m_t,
                 step_size);

    if (m_order == 1) {
      // Euler method (1st order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, params);
      auto &k1 = *asCudmState(k1State);
      k1 *= step_size;
      castSimState += k1;
    } else if (m_order == 2) {
      // Midpoint method (2nd order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, params);
      auto &k1 = *asCudmState(k1State);
      k1 *= (step_size / 2.0);

      castSimState += k1;
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] =
            m_schedule.get_value_function()(param, m_t + step_size / 2.0);
      }
      auto k2State = m_stepper->compute(*m_state, m_t + step_size / 2.0,
                                        step_size, params);
      auto &k2 = *asCudmState(k2State);
      k2 *= (step_size / 2.0);

      castSimState += k2;
    } else if (m_order == 4) {
      // Runge-Kutta method (4th order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, params);
      auto &k1 = *asCudmState(k1State);
      CuDensityMatState rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp += (k1 * (step_size / 2));

      for (const auto &param : m_schedule.get_parameters()) {
        params[param] =
            m_schedule.get_value_function()(param, m_t + step_size / 2.0);
      }
      auto k2State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp))),
          m_t + step_size / 2.0, step_size, params);
      auto &k2 = *asCudmState(k2State);
      CuDensityMatState rho_temp_2 = CuDensityMatState::clone(castSimState);
      rho_temp_2 += (k2 * (step_size / 2));

      auto k3State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp_2))),
          m_t + step_size / 2.0, step_size, params);
      auto &k3 = *asCudmState(k3State);
      CuDensityMatState rho_temp_3 = CuDensityMatState::clone(castSimState);
      rho_temp_3 += (k3 * step_size);

      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t + step_size);
      }
      auto k4State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp_3))),
          m_t + step_size, step_size, params);
      auto &k4 = *asCudmState(k4State);
      castSimState += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    // Update time
    m_t += step_size;
  }
}
} // namespace integrators
} // namespace cudaq
