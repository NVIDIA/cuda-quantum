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
  if (!this->stepper) {
    throw std::runtime_error("Time stepper is not initialized.");
  }

  if (this->integrator_options.find("dt") == this->integrator_options.end()) {
    throw std::invalid_argument(
        "Time step size (dt) is missing from integrator options.");
  }

  double dt = this->integrator_options["dt"];
  if (dt <= 0) {
    throw std::invalid_argument("Invalid time step size for integration.");
  }

  while (this->t < target_time) {
    double step_size = std::min(dt, target_time - this->t);

    // std::cout << "Runge-Kutta step at time " << this->t
    //           << " with step size: " << step_size << std::endl;

    if (this->substeps_ == 1) {
      // Euler method (1st order)
      cudm_state k1 = this->stepper->compute(this->state, this->t, step_size);
      k1 *= step_size;
      this->state += k1;
    } else if (this->substeps_ == 2) {
      // Midpoint method (2nd order)
      cudm_state k1 = this->stepper->compute(this->state, this->t, step_size);
      k1 *= (step_size / 2.0);

      this->state += k1;

      cudm_state k2 = this->stepper->compute(
          this->state, this->t + step_size / 2.0, step_size);
      k2 *= (step_size / 2.0);

      this->state += k2;
    } else if (this->substeps_ == 4) {
      // Runge-Kutta method (4th order)
      cudm_state k1 = this->stepper->compute(this->state, this->t, step_size);
      k1 *= step_size;

      this->state += k1 * 0.5;

      cudm_state k2 = this->stepper->compute(
          this->state, this->t + step_size / 2.0, step_size);
      k2 *= step_size;

      this->state += k2 * 0.5;

      cudm_state k3 = this->stepper->compute(
          this->state, this->t + step_size / 2.0, step_size);
      k3 *= step_size;

      this->state += k3;

      cudm_state k4 =
          this->stepper->compute(this->state, this->t + step_size, step_size);
      k4 *= step_size;

      this->state += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0);
    }

    // Update time
    this->t += step_size;
  }

  std::cout << "Integration complete. Final time: " << this->t << std::endl;
}

} // namespace cudaq
