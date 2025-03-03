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
#include "cudaq/dynamics_integrators.h"

namespace cudaq {

cudaq::Schedule createDummySchedule() {
  std::vector<std::complex<double>> dummy_steps = {{0.0, 0.0}};
  std::vector<std::string> dummy_params = {"t"};
  return cudaq::Schedule(
      dummy_steps, dummy_params,
      [](const std::string &, const std::complex<double> &val) { return val; });
}

RungeKuttaIntegrator::RungeKuttaIntegrator()
    : m_t(0.0), m_schedule(createDummySchedule()) {}

void RungeKuttaIntegrator::setSystem(const SystemDynamics &system,
                                     const cudaq::Schedule &schedule) {
  m_system = system;
  m_schedule = schedule;
  m_stepper.reset();
}

void RungeKuttaIntegrator::setState(cudaq::state initial_state, double t0) {
  m_state = std::make_shared<cudaq::state>(initial_state);
  m_t = t0;
}

std::pair<double, cudaq::state> RungeKuttaIntegrator::getState() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");

  auto cudmState =
      new CuDensityMatState(castSimState->get_handle(), *castSimState,
                            castSimState->get_hilbert_space_dims());

  return std::make_pair(m_t, cudaq::state(cudmState));
}

void RungeKuttaIntegrator::integrate(double targetTime) {
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
            .constructLiouvillian(*m_system.hamiltonian, m_system.collapseOps,
                                  m_system.modeExtents, params,
                                  castSimState.is_density_matrix());
    m_stepper = std::make_unique<CuDensityMatTimeStepper>(
        castSimState.get_handle(), liouvillian);
  }
  const auto substeps = order.value_or(4);
  while (m_t < targetTime) {
    double step_size =
        std::min(dt.value_or(targetTime - m_t), targetTime - m_t);

    // std::cout << "Runge-Kutta step at time " << m_t
    //           << " with step size: " << step_size << std::endl;

    if (substeps == 1) {
      // Euler method (1st order)
      for (const auto &param : m_schedule.get_parameters()) {
        params[param] = m_schedule.get_value_function()(param, m_t);
      }
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, params);
      auto &k1 = *asCudmState(k1State);
      // k1.dump(std::cout);
      k1 *= step_size;
      castSimState += k1;
    } else if (substeps == 2) {
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
    } else if (substeps == 4) {
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
} // namespace cudaq
