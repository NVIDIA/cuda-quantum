/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/base_integrator.h"
#include "cudaq/cudm_state.h"
#include "cudaq/cudm_time_stepper.h"
#include <iostream>
#include <memory>

namespace cudaq {
class runge_kutta_integrator : public BaseIntegrator<cudm_state> {
public:
  /// @brief Constructor to initialize the Runge-Kutta integrator
  /// @param initial_state Initial quantum state.
  /// @param t0 Initial time.
  /// @param stepper Time stepper instance.
  /// @param substeps Number of Runge-Kutta substeps (must be 1, 2, or 4)
  runge_kutta_integrator(cudm_state &&initial_state, double t0,
                         std::shared_ptr<cudm_time_stepper> stepper,
                         int substeps = 4)
      : BaseIntegrator<cudm_state>(std::move(initial_state), t0, stepper),
        substeps_(substeps) {
    if (!stepper) {
      throw std::invalid_argument("Time stepper must be initialized.");
    }

    if (substeps_ != 1 && substeps_ != 2 && substeps_ != 4) {
      throw std::invalid_argument("Runge-Kutta substeps must be 1, 2, or 4.");
    }
    this->post_init();
  }

  /// @brief Perform Runge-Kutta integration until the target time.
  /// @param target_time The final time to integrate to.
  void integrate(double target_time) override;

protected:
  /// @brief Any post-initialization setup
  void post_init() override {}

private:
  // Number of substeps in RK integration (1, 2, or 4)
  int substeps_;
};
} // namespace cudaq
