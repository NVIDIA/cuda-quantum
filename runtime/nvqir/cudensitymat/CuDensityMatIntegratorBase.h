/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuDensityMatContext.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "cudaq/algorithms/base_integrator.h"

namespace cudaq {

/// @brief Internal helpers shared by all cuDensityMat-backed integrators.
struct CuDensityMatIntegratorHelper {

  /// @brief Cast a cudaq::state to CuDensityMatState*, throwing on failure.
  static CuDensityMatState *asCudmState(cudaq::state &cudaqState) {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  }

  /// @brief Common setState implementation for all cuDensityMat integrators.
  static void setState(std::shared_ptr<cudaq::state> &m_state, double &m_t,
                       const cudaq::state &initialState, double t0) {
    auto *cudmState = asCudmState(*const_cast<cudaq::state *>(&initialState));
    m_state = std::make_shared<cudaq::state>(
        CuDensityMatState::clone(*cudmState).release());
    m_t = t0;
  }

  /// @brief Common getState implementation for all cuDensityMat integrators.
  static std::pair<double, cudaq::state>
  getState(std::shared_ptr<cudaq::state> &m_state, double m_t) {
    auto *castSimState = asCudmState(*m_state);
    return std::make_pair(
        m_t, cudaq::state(CuDensityMatState::clone(*castSimState).release()));
  }

  /// @brief Compute the next sub-step size toward targetTime, respecting m_dt.
  static double computeStepSize(double m_t, double targetTime,
                                const std::optional<double> &m_dt) {
    return std::min(m_dt.value_or(targetTime - m_t), targetTime - m_t);
  }

  /// @brief Lazily construct the time stepper from the system and schedule.
  ///
  /// Must be called at the start of integrate() before the time-stepping loop.
  static void ensureStepper(std::unique_ptr<base_time_stepper> &m_stepper,
                             std::shared_ptr<cudaq::state> &m_state,
                             const SystemDynamics &m_system,
                             const cudaq::schedule &m_schedule) {
    if (m_stepper)
      return;
    auto &castSimState = *asCudmState(*m_state);
    std::unordered_map<std::string, std::complex<double>> params;
    for (const auto &param : m_schedule.get_parameters())
      params[param] = m_schedule.get_value_function()(param, 0.0);

    auto liouvillian =
        m_system.superOp.has_value()
            ? cudaq::dynamics::Context::getCurrentContext()
                  ->getOpConverter()
                  .constructLiouvillian({m_system.superOp.value()},
                                        m_system.modeExtents, params)
            : cudaq::dynamics::Context::getCurrentContext()
                  ->getOpConverter()
                  .constructLiouvillian({m_system.hamiltonian},
                                        {m_system.collapseOps},
                                        m_system.modeExtents, params,
                                        castSimState.is_density_matrix());
    m_stepper = std::make_unique<CuDensityMatTimeStepper>(
        castSimState.get_handle(), liouvillian);
  }

  /// @brief Evaluate all schedule parameters at time t.
  static std::unordered_map<std::string, std::complex<double>>
  scheduleParamsAt(const cudaq::schedule &m_schedule, double t) {
    std::unordered_map<std::string, std::complex<double>> params;
    for (const auto &param : m_schedule.get_parameters())
      params[param] = m_schedule.get_value_function()(param, t);
    return params;
  }
};

} // namespace cudaq
