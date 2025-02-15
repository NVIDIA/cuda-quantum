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
class runge_kutta : public BaseIntegrator {

public:
  std::optional<int> order;
  std::optional<double> dt;

public:
  runge_kutta() = default;
  // TODO
  void integrate(double target_time) override {}
  void set_state(cudaq::state initial_state, double t0) override {}
  std::pair<double, cudaq::state> get_state() override {
    return std::make_pair(m_t, cudaq::state(nullptr));
  }

private:
  double m_t;
};
} // namespace cudaq
