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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

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

  /// @brief Number of equal sub-steps covering `[currentTime, targetTime]`.
  /// Zero if the target is already reached.
  ///
  /// A sub-step may run up to one `ulp` of the timestamps past `maxStepSize`,
  /// which is the price of not splitting an interval that is only a rounding
  /// error wider than a whole number of steps.
  static std::int64_t subStepCount(double currentTime, double targetTime,
                                   const std::optional<double> &maxStepSize) {
    const auto remaining = targetTime - currentTime;
    if (remaining <= 0.0)
      return 0;
    if (!maxStepSize.has_value() || *maxStepSize <= 0.0 ||
        remaining <= *maxStepSize)
      return 1;
    // `remaining` is a difference of two rounded timestamps, so an interval
    // that nominally holds N whole steps can measure up to an ulp longer.
    const auto remainingRoundingError =
        std::numeric_limits<double>::epsilon() *
        std::max(std::abs(currentTime), std::abs(targetTime));
    // Without this, ceil() would answer N + 1 and spend a whole extra sub-step
    // covering that ulp.
    const auto count =
        std::ceil((remaining - remainingRoundingError) / *maxStepSize);
    // Refuse rather than let the conversion below overflow, which would yield a
    // garbage count and silently ignore `maxStepSize`. Also rejects NaN.
    if (!(count <=
          static_cast<double>(std::numeric_limits<std::int64_t>::max())))
      throw std::invalid_argument(
          "max_step_size is too small to cover the integration interval.");
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(count));
  }

  /// @brief End time of sub-step `index` (1-based) out of `count`.
  ///
  /// Callers must derive the step size as `subStepTime(...) - currentTime`, so
  /// that the time advance equals the propagated step by construction.
  static double subStepTime(double startTime, double targetTime,
                            std::int64_t index, std::int64_t count) {
    // Interpolate rather than accumulate: no drift, and the last index is the
    // target itself, so the schedule point is hit exactly without a tolerance.
    return index >= count ? targetTime
                          : startTime + (targetTime - startTime) *
                                            (static_cast<double>(index) /
                                             static_cast<double>(count));
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
