/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatIntegratorBase.h"
#include "CuDensityMatUtils.h"
#include "support/error_norm.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/cudaq_mpi.h"
#include "cudaq/runtime/logger/logger.h"
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <stdexcept>
#include <vector>

namespace cudaq::integrators {

// Dormand-Prince RK5(4) adaptive integrator.
// Reference: Dormand, J. R.; Prince, P. J. (1980), "A family of embedded
// Runge-Kutta formulae", Journal of Computational and Applied Mathematics.
//
// Uses the mainlined cuDensityMat time stepper to evaluate the Liouvillian
// action f(t, y) = L(t) y, forms the embedded 5th- and 4th-order solutions,
// and adapts the step size from the scaled error between them.

using cudmIntHelp = CuDensityMatIntegratorHelper;

namespace {
// Butcher tableau (nodes).
constexpr std::array<double, 7> kNodes = {0.0,       0.2, 0.3, 0.8,
                                          8.0 / 9.0, 1.0, 1.0};

// Lower-triangular a_{ij} coefficients.
constexpr std::array<std::array<double, 6>, 7> kA = {
    {{0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
     {0.2, 0.0, 0.0, 0.0, 0.0, 0.0},
     {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0},
     {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0},
     {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0,
      0.0, 0.0},
     {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,
      -5103.0 / 18656.0, 0.0},
     {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
      11.0 / 84.0}}};

// 5th-order solution weights.
constexpr std::array<double, 7> kB5 = {
    35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0,
    11.0 / 84.0,  0.0};

// Embedded 4th-order solution weights.
constexpr std::array<double, 7> kB4 = {5179.0 / 57600.0,    0.0,
                                       7571.0 / 16695.0,    393.0 / 640.0,
                                       -92097.0 / 339200.0, 187.0 / 2100.0,
                                       1.0 / 40.0};

// Step-size controller parameters.
constexpr double kSafetyFactor = 0.9;
constexpr double kMaxScale = 5.0; // Max step-size growth per accepted step.
constexpr double kMinScale = 0.2; // Max step-size shrink per rejected step.
constexpr double kOrder = 5.0;    // Method order used for step adaptation.

/// @brief Compute the scaled RMS error norm between the embedded solutions.
///
/// Returns a dimensionless error scaled so that a value <= 1 means the step
/// meets the requested (rtol, atol) tolerance. Uses a scipy-style element-wise
/// scale (atol + rtol * max(|y5_i|, |y4_i|)) and reduces on the device to avoid
/// per-step device-to-host copies of the full state. For distributed
/// (multi-GPU) states the local partial sums are all-reduced across ranks so
/// the norm is over the global state.
double computeErrorNorm(const CuDensityMatState &y5,
                        const CuDensityMatState &y4, double rtol, double atol) {
  const std::size_t nLocal = y5.getTensor().get_num_elements();

  double sumsqLocal = 0.0;
  cudaq::detail::scaled_error_sumsq(y5.get_device_pointer(),
                                    y4.get_device_pointer(), nLocal, rtol, atol,
                                    &sumsqLocal);

  double sumsqGlobal = sumsqLocal;
  double nGlobal = static_cast<double>(nLocal);
  // Distributed density-matrix / state-vector shards: combine the per-rank
  // partial sums and element counts so the RMS norm is over the full state.
  if (cudaq::mpi::is_initialized() && cudaq::mpi::num_ranks() > 1) {
    sumsqGlobal = cudaq::mpi::all_reduce(sumsqLocal, std::plus<double>());
    nGlobal = cudaq::mpi::all_reduce(static_cast<double>(nLocal),
                                     std::plus<double>());
  }

  if (nGlobal == 0.0)
    return 0.0;
  return std::sqrt(sumsqGlobal / nGlobal);
}

/// @brief Select the next step size from the scaled error estimate.
double adaptStepSize(double errorNorm, double dtCurrent, double dtMin,
                     double dtMax) {
  double factor;
  if (errorNorm == 0.0) {
    factor = kMaxScale;
  } else {
    factor = kSafetyFactor * std::pow(1.0 / errorNorm, 1.0 / kOrder);
    factor = std::max(kMinScale, std::min(kMaxScale, factor));
  }
  return std::max(dtMin, std::min(dtMax, dtCurrent * factor));
}
} // namespace

dopri5::dopri5(double rtol, double atol, double dt_initial, double dt_min,
               double dt_max)
    : m_rtol(rtol), m_atol(atol), m_dt(dt_initial), m_dt_min(dt_min),
      m_dt_max(dt_max), m_t(0.0) {
  if (rtol <= 0.0 || atol <= 0.0)
    throw std::invalid_argument(
        "dopri5 integrator requires positive rtol and atol.");
  if (dt_min <= 0.0 || dt_max <= 0.0 || dt_min > dt_max)
    throw std::invalid_argument(
        "dopri5 integrator requires 0 < dt_min <= dt_max.");
}

std::shared_ptr<base_integrator> dopri5::clone() {
  auto cloned = std::make_shared<cudaq::integrators::dopri5>(
      m_rtol, m_atol, m_dt, m_dt_min, m_dt_max);
  cloned->m_t = this->m_t;
  // Deep-copy the state so the clone owns a distinct device buffer and the two
  // integrators evolve fully independently.
  if (m_state) {
    auto *cudm = cudmIntHelp::asCudmState(*m_state);
    cloned->m_state = std::make_shared<cudaq::state>(
        CuDensityMatState::clone(*cudm).release());
  }
  cloned->m_system = this->m_system;
  cloned->m_schedule = this->m_schedule;
  cloned->m_stats = this->m_stats;
  return cloned;
}

void dopri5::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
  // A new initial state invalidates the cached FSAL stage.
  m_fsalK1.reset();
  resetStats();
}

std::pair<double, cudaq::state> dopri5::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

void dopri5::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("dopri5::integrate");
  cudmIntHelp::ensureStepper(m_stepper, m_state, m_system, m_schedule);

  // Guard against runaway step rejection (e.g. dt driven to dt_min).
  constexpr std::size_t MAX_ITERATIONS = 100000;
  std::size_t iterations = 0;

  while (m_t < targetTime) {
    if (++iterations > MAX_ITERATIONS)
      throw std::runtime_error(
          "dopri5 integrator exceeded maximum iterations; possible "
          "convergence issue or step size driven below dt_min.");

    const double dt = std::min(m_dt, targetTime - m_t);
    auto &y = *cudmIntHelp::asCudmState(*m_state);

    // Evaluate the seven Dormand-Prince stages. Stage j uses the state
    // y + dt * sum_{i<j} a[j][i] k_i, sampled at time t + c[j] dt.
    std::array<std::shared_ptr<cudaq::state>, 7> kStates;
    auto evalStage = [&](int j, CuDensityMatState &stageInput) {
      auto params =
          cudmIntHelp::scheduleParamsAt(m_schedule, m_t + kNodes[j] * dt);
      cudaq::state stageState(CuDensityMatState::clone(stageInput).release());
      auto result =
          m_stepper->compute(stageState, m_t + kNodes[j] * dt, params);
      kStates[j] = std::make_shared<cudaq::state>(std::move(result));
    };

    // k1 = f(t, y). Reuse the FSAL stage (k7 of the previous accepted step, or
    // the k1 of a just-rejected step) when available: both equal f at the
    // current (t, y), so we skip one Liouvillian evaluation.
    if (m_fsalK1)
      kStates[0] = m_fsalK1;
    else
      evalStage(0, y);
    auto stageK = [&](int j) -> CuDensityMatState & {
      return *cudmIntHelp::asCudmState(*kStates[j]);
    };

    for (int j = 1; j < 7; ++j) {
      auto stageInput = CuDensityMatState::clone(y);
      for (int i = 0; i < j; ++i)
        if (kA[j][i] != 0.0)
          stageInput->accumulate_inplace(stageK(i), dt * kA[j][i]);
      evalStage(j, *stageInput);
    }

    // Embedded 5th- and 4th-order solutions.
    auto y5 = CuDensityMatState::clone(y);
    auto y4 = CuDensityMatState::clone(y);
    for (int j = 0; j < 7; ++j) {
      if (kB5[j] != 0.0)
        y5->accumulate_inplace(stageK(j), dt * kB5[j]);
      if (kB4[j] != 0.0)
        y4->accumulate_inplace(stageK(j), dt * kB4[j]);
    }

    const double errorNorm = computeErrorNorm(*y5, *y4, m_rtol, m_atol);
    const double dtNext = adaptStepSize(errorNorm, dt, m_dt_min, m_dt_max);
    const bool accept = (errorNorm <= 1.0);

    if (accept) {
      m_state = std::make_shared<cudaq::state>(y5.release());
      m_t += dt;
      m_dt = dtNext;
      // FSAL: k7 == f(t + dt, y5) is the next step's k1 = f(t_new, y_new).
      m_fsalK1 = kStates[6];
      ++m_stats.accepted_steps;
      m_stats.min_dt_used = std::min(m_stats.min_dt_used, dt);
      m_stats.max_dt_used = std::max(m_stats.max_dt_used, dt);
      m_stats.avg_dt = (m_stats.avg_dt * (m_stats.accepted_steps - 1) + dt) /
                       m_stats.accepted_steps;
    } else {
      m_dt = dtNext;
      // t and y are unchanged, so k1 = f(t, y) stays valid for the retry.
      m_fsalK1 = kStates[0];
      ++m_stats.rejected_steps;
    }
  }
}

} // namespace cudaq::integrators
