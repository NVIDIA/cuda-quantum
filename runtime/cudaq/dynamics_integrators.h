/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/base_integrator.h"
#include "cudaq/base_time_stepper.h"
#include "cudaq/operators.h"
#include <memory>

namespace cudaq {
namespace integrators {

class runge_kutta : public cudaq::base_integrator {

public:
  std::optional<int> order;
  std::optional<double> dt;

public:
  runge_kutta();
  void integrate(double targetTime) override;
  void setState(const cudaq::state &initialState, double t0) override;
  std::pair<double, cudaq::state> getState() override;
  std::shared_ptr<base_integrator> clone() override;

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
};
} // namespace integrators
} // namespace cudaq
