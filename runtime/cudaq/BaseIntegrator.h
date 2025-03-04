/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "operators.h"
#include "schedule.h"
#include <map>
#include <memory>
#include <vector>

namespace cudaq {
// Struct captures the system dynamics needed by the integrator
struct SystemDynamics {
  std::vector<int64_t> modeExtents;
  operator_sum<cudaq::matrix_operator> hamiltonian;
  std::vector<operator_sum<cudaq::matrix_operator>> collapseOps;
  std::unordered_map<std::string, std::complex<double>> parameters;

  SystemDynamics(
      const std::vector<int64_t> extents,
      const operator_sum<cudaq::matrix_operator> &ham,
      const std::vector<operator_sum<cudaq::matrix_operator>> &cOps = {},
      const std::unordered_map<std::string, std::complex<double>> &params = {})
      : modeExtents(extents), hamiltonian(ham), collapseOps(cOps),
        parameters(params) {}

  SystemDynamics()
      : hamiltonian(operator_sum<cudaq::matrix_operator>(
            cudaq::matrix_operator::empty())){};
};
class BaseIntegrator {
public:
  /// @brief Default constructor
  BaseIntegrator() = default;

  virtual ~BaseIntegrator() = default;

  /// @brief Set the initial state and time
  virtual void setState(const cudaq::state &initialState, double t0) = 0;
  /// @brief Set the system dynamics
  virtual void setSystem(const cudaq::SystemDynamics &system,
                         const cudaq::Schedule &schedule) = 0;
  /// @brief Perform integration to the target time.
  virtual void integrate(double targetTime) = 0;

  /// @brief Get the current time and state.
  virtual std::pair<double, cudaq::state> getState() = 0;

  /// @brief Create a clone of this integrator.
  virtual std::shared_ptr<BaseIntegrator> clone() = 0;
};
} // namespace cudaq
