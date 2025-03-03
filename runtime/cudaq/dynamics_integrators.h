/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/BaseIntegrator.h"
#include "cudaq/BaseTimeStepper.h"
#include "cudaq/operators.h"
#include <memory>

namespace cudaq {
struct SystemDynamics {
  operator_sum<cudaq::matrix_operator> *hamiltonian = nullptr;
  std::vector<operator_sum<cudaq::matrix_operator>> collapseOps;
  std::vector<int64_t> modeExtents;
  std::unordered_map<std::string, std::complex<double>> parameters;
};

class RungeKuttaIntegrator : public BaseIntegrator {

public:
  std::optional<int> order;
  std::optional<double> dt;

public:
  RungeKuttaIntegrator();
  void integrate(double targetTime) override;
  void setState(cudaq::state initialState, double t0) override;
  std::pair<double, cudaq::state> getState() override;
  void setSystem(const SystemDynamics &system, const cudaq::Schedule &schedule);

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  SystemDynamics m_system;
  std::unique_ptr<BaseTimeStepper> m_stepper;
  cudaq::Schedule m_schedule;
};
} // namespace cudaq
