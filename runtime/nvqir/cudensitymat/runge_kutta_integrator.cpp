/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runge_kutta_integrator.h"
#include "cudm_state.h"
#include "cudm_time_stepper.h"
namespace cudaq {
void runge_kutta_integrator::set_state(cudaq::state initial_state, double t0) {
  // TODO
}
std::pair<double, cudaq::state> runge_kutta_integrator::get_state() {
  // TODO:
  return std::make_pair(0.0, cudaq::state(nullptr));
}

// FIXME: remove this
std::pair<double, cudm_state *> runge_kutta_integrator::get_cudm_state() {
  return std::make_pair(m_t, m_state.get());
}

void runge_kutta_integrator::integrate(double target_time) {
  if (!m_stepper) {
    throw std::runtime_error("Time stepper is not initialized.");
  }

  if (dt.has_value() && dt.value() <= 0.0) {
    throw std::invalid_argument("Invalid time step size for integration.");
  }

  if (!m_state) {
    throw std::runtime_error("Initial state has not been set.");
  }
  const auto substeps = order.value_or(4);
  while (m_t < target_time) {
    double step_size =
        std::min(dt.value_or(target_time - m_t), target_time - m_t);

    // std::cout << "Runge-Kutta step at time " << m_t
    //           << " with step size: " << step_size << std::endl;

    if (substeps == 1) {
      // Euler method (1st order)
      cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);
      k1 *= step_size;
      *m_state += k1;
    } else if (substeps == 2) {
      // Midpoint method (2nd order)
      cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);
      k1 *= (step_size / 2.0);

      *m_state += k1;

      cudm_state k2 =
          m_stepper->compute(*m_state, m_t + step_size / 2.0, step_size);
      k2 *= (step_size / 2.0);

      *m_state += k2;
    } else if (substeps == 4) {
      // Runge-Kutta method (4th order)
      cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);

      cudm_state rho_temp = cudm_state::clone(*m_state);
      rho_temp += (k1 * (step_size / 2));

      cudm_state k2 =
          m_stepper->compute(rho_temp, m_t + step_size / 2.0, step_size);

      cudm_state rho_temp_2 = cudm_state::clone(*m_state);
      rho_temp_2 += (k2 * (step_size / 2));

      cudm_state k3 =
          m_stepper->compute(rho_temp_2, m_t + step_size / 2.0, step_size);

      cudm_state rho_temp_3 = cudm_state::clone(*m_state);
      rho_temp_3 += (k3 * step_size);

      cudm_state k4 =
          m_stepper->compute(rho_temp_3, m_t + step_size, step_size);

      *m_state += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    // Update time
    m_t += step_size;
  }

  // std::cout << "Integration complete. Final time: " << m_t << std::endl;
}

// TODO: remove this
runge_kutta_integrator::runge_kutta_integrator(
    cudm_state &&initial_state, double t0,
    std::shared_ptr<cudm_time_stepper> stepper, int substeps)
    : m_t(t0), m_stepper(stepper), order(substeps) {

  m_state = std::make_unique<cudm_state>(std::move(initial_state));
}
void runge_kutta_integrator::set_stepper(
    std::shared_ptr<cudm_time_stepper> stepper) {
  m_stepper = stepper;
}

void runge_kutta_integrator::set_state(cudm_state &&initial_state) {
  m_state = std::make_unique<cudm_state>(std::move(initial_state));
  m_t = 0.0;
}
} // namespace cudaq
