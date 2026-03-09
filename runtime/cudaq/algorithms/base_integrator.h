/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/operators.h"
#include "cudaq/schedule.h"
#include <map>
#include <memory>
#include <vector>

namespace cudaq {
// Struct captures the system dynamics needed by the integrator
struct SystemDynamics {
  std::vector<std::int64_t> modeExtents;
  std::vector<sum_op<cudaq::matrix_handler>> hamiltonian;
  std::vector<std::vector<sum_op<cudaq::matrix_handler>>> collapseOps;
  std::optional<std::vector<super_op>> superOp;
  std::unordered_map<std::string, std::complex<double>> parameters;

  SystemDynamics(
      const std::vector<std::int64_t> extents,
      const sum_op<cudaq::matrix_handler> &ham,
      const std::vector<sum_op<cudaq::matrix_handler>> &cOps = {},
      const std::unordered_map<std::string, std::complex<double>> &params = {})
      : modeExtents(extents), hamiltonian({ham}), collapseOps({cOps}),
        parameters(params) {}
  SystemDynamics(const std::vector<std::int64_t> extents,
                 const super_op &superOperator)
      : modeExtents(extents), superOp({superOperator}) {}
  SystemDynamics(
      const std::vector<std::int64_t> extents,
      const std::vector<sum_op<cudaq::matrix_handler>> &ham,
      const std::vector<std::vector<sum_op<cudaq::matrix_handler>>> &cOps = {},
      const std::unordered_map<std::string, std::complex<double>> &params = {})
      : modeExtents(extents), hamiltonian(ham), collapseOps(cOps),
        parameters(params) {}
  SystemDynamics(const std::vector<std::int64_t> extents,
                 const std::vector<super_op> &superOperator)
      : modeExtents(extents), superOp(superOperator) {}
  SystemDynamics() : hamiltonian({cudaq::matrix_op::empty()}){};
};

class base_time_stepper;
class base_integrator {
public:
  /// @brief Default constructor
  base_integrator() = default;

  virtual ~base_integrator() = default;

  /// @brief Set the initial state and time
  virtual void setState(const cudaq::state &initialState, double t0) = 0;

  /// @brief Perform integration to the target time.
  virtual void integrate(double targetTime) = 0;

  /// @brief Get the current time and state.
  virtual std::pair<double, cudaq::state> getState() = 0;

  /// @brief Create a clone of this integrator.
  // e.g., cloning an integrator to keep inside async. functors.
  virtual std::shared_ptr<base_integrator> clone() = 0;

protected:
  friend class integrator_helper;
  SystemDynamics m_system;
  cudaq::schedule m_schedule;
  std::unique_ptr<base_time_stepper> m_stepper;
};

class integrator_helper {
public:
  static void init_system_dynamics(base_integrator &integrator,
                                   const SystemDynamics &system,
                                   const cudaq::schedule &schedule) {
    integrator.m_system = system;
    integrator.m_schedule = schedule;
    integrator.m_stepper.reset();
  }

  static void init_system_dynamics(base_integrator &integrator,
                                   const std::vector<super_op> &superOps,
                                   std::vector<std::int64_t> modeExtents,
                                   const cudaq::schedule &schedule) {
    SystemDynamics systemDynamics(modeExtents, superOps);
    integrator.m_system = systemDynamics;
    integrator.m_schedule = schedule;
    integrator.m_stepper.reset();
  }
};
} // namespace cudaq
