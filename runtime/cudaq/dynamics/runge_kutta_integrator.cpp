/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runge_kutta_integrator.h"
#include <iostream>

using namespace cudaq;

namespace cudaq {
void runge_kutta_integrator::integrate(double target_time) {
  if (!stepper) {
    throw std::runtime_error("Time stepper is not initialized.");
  }

  if (integrator_options.find("dt") == integrator_options.end()) {
    throw std::invalid_argument(
        "Time step size (dt) is missing from integrator options.");
  }

  double dt = integrator_options["dt"];
  if (dt <= 0) {
    throw std::invalid_argument("Invalid time step size for integration.");
  }

  while (t < target_time) {
    double step_size = std::min(dt, target_time - t);

    std::cout << "Runge-Kutta step at time " << t
              << " with step size: " << step_size << std::endl;

    if (substeps_ == 1) {
      // Euler method (1st order)
      cudm_state k1 = stepper->compute(state, t, step_size);
      state += k1;
    } else if (substeps_ == 2) {
      // Midpoint method (2nd order)
      cudm_state k1 = stepper->compute(state, t, step_size / 2.0);
      cudm_state k2 = stepper->compute(k1, t + step_size / 2.0, step_size);
      state += (k1 + k2) * 0.5;
    } else if (substeps_ == 4) {
      // Runge-Kutta method (4th order)
      cudm_state k1 = stepper->compute(state, t, step_size / 2.0);
      cudm_state k2 =
          stepper->compute(k1, t + step_size / 2.0, step_size / 2.0);
      cudm_state k3 = stepper->compute(k2, t + step_size / 2.0, step_size);
      cudm_state k4 = stepper->compute(k3, t + step_size, step_size);
      state += (k1 + (k2 + k3) * 2.0 + k4) * (1.0 / 6.0);
    }

    // Update time
    t += step_size;
  }

  std::cout << "Integration complete. Final time: " << t << std::endl;
}
} // namespace cudaq
