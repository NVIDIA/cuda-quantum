/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/base_integrator.h"
#include <memory>

namespace cudaq {
class cudm_state;
class cudm_time_stepper;
class runge_kutta_integrator : public BaseIntegrator {

public:
  std::optional<int> order;
  std::optional<double> dt;

public:
  runge_kutta_integrator() = default;
  runge_kutta_integrator(cudm_state &&initial_state, double t0,
                         std::shared_ptr<cudm_time_stepper> stepper,
                         int substeps = 4);

  void integrate(double target_time) override;
  void set_state(cudaq::state initial_state, double t0) override;
  void set_state(cudm_state &&initial_state);
  void set_stepper(std::shared_ptr<cudm_time_stepper> stepper);
  std::pair<double, cudaq::state> get_state() override;
  std::pair<double, cudm_state*> get_cudm_state();
private:
  std::unique_ptr<cudm_state> m_state;
  double m_t;
  std::shared_ptr<cudm_time_stepper> m_stepper;
};
} // namespace cudaq
