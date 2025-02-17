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
struct SystemDynamics {
  operator_sum<cudaq::matrix_operator> *hamiltonian = nullptr;
  std::vector<operator_sum<cudaq::matrix_operator> *> collapseOps;
  std::vector<int64_t> modeExtents;
  std::unordered_map<std::string, std::complex<double>> parameters;
};

class runge_kutta : public BaseIntegrator {

public:
  std::optional<int> order;
  std::optional<double> dt;

public:
  runge_kutta() = default;
  void integrate(double target_time) override;
  void set_state(cudaq::state initial_state, double t0) override;
  std::pair<double, cudaq::state> get_state() override;
  void set_system(const SystemDynamics &system);

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  SystemDynamics m_system;
  std::unique_ptr<TimeStepper> m_stepper;
};
} // namespace cudaq
