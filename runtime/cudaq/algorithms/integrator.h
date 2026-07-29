/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/base_integrator.h"
#include "cudaq/algorithms/base_time_stepper.h"
#include "cudaq/operators.h"
#include <memory>

namespace cudaq {
namespace integrators {

class runge_kutta : public cudaq::base_integrator {
public:
  /// @brief The default `Runge-Kutta` integration order.
  // Note: we use 4th order as the default since (1) it produces better
  // convergence/stability and (2) that is the typical order that the name
  // `Runge-Kutta` is associated with (e.g., RK method can be generalized to
  // cover 1st order Euler method)
  static constexpr int default_order = 4;
  /// @brief Constructor
  // (1) Integration order
  // (2) Max step size: if none provided, the schedule of time points where we
  // want to compute and save intermediate results will be used. If provided,
  // the integrator will make sub-steps no larger than this value to integrate
  // toward scheduled time points.
  runge_kutta(int order = default_order,
              const std::optional<double> &max_step_size = {});
  /// @brief Integrate toward a specified time point.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current state of the integrator
  // Returns the current time point and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator.
  std::shared_ptr<base_integrator> clone() override;

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  int m_order;
  std::optional<double> m_dt;
};

/// @brief `Crank-Nicolson` integrator for quantum state evolution.
class crank_nicolson : public cudaq::base_integrator {
public:
  /// @brief Default number of predictor-corrector iterations.
  static constexpr int default_num_corrector_steps = 2;

  /// @brief Constructor.
  /// @param num_corrector_steps Number of corrector iterations per step.
  ///        (default: 2, i.e., one predictor + two `correctors`)
  /// @param max_step_size Optional maximum internal sub-step size. When
  ///        provided the integrator will sub-step by at most this amount
  ///        when integrating toward each scheduled time point.
  crank_nicolson(int num_corrector_steps = default_num_corrector_steps,
                 const std::optional<double> &max_step_size = {});

  /// @brief Integrate toward a specified time point.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration.
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current time and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator.
  std::shared_ptr<base_integrator> clone() override;

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  int m_num_corrector_steps;
  std::optional<double> m_dt;
};

/// @brief `Magnus` expansion integrator for quantum state evolution.
class magnus_expansion : public cudaq::base_integrator {
public:
  /// @brief Default maximum number of Taylor series terms.
  static constexpr int default_num_taylor_terms = 10;

  /// @brief Constructor.
  /// @param num_taylor_terms Maximum number of Taylor terms used to
  ///        approximate exp(h·L_mid). (default: 10; the series exits
  ///        early when a term is negligibly small.)
  /// @param max_step_size Optional maximum internal sub-step size.
  magnus_expansion(int num_taylor_terms = default_num_taylor_terms,
                   const std::optional<double> &max_step_size = {});

  /// @brief Integrate toward a specified time point.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration.
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current time and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator.
  std::shared_ptr<base_integrator> clone() override;

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  int m_num_taylor_terms;
  std::optional<double> m_dt;
};

/// @brief Dormand-Prince RK5(4) adaptive-timestep integrator (`dopri5`).
///
/// GPU-accelerated adaptive integrator using the Dormand-Prince embedded
/// RK5(4) pair with an error-controlled step-size selector. Well suited to
/// stiff, high-frequency driven (transmon) dynamics where a fixed step size
/// is either inefficient (too small everywhere) or inaccurate (too large
/// across fast transients).
class dopri5 : public cudaq::base_integrator {
public:
  /// @brief Default relative tolerance for the embedded error estimate.
  static constexpr double default_rtol = 1e-6;
  /// @brief Default absolute tolerance for the embedded error estimate.
  static constexpr double default_atol = 1e-8;

  /// @brief Constructor.
  /// @param rtol Relative tolerance for the embedded RK5(4) error estimate.
  /// @param atol Absolute tolerance for the embedded RK5(4) error estimate.
  /// @param dt_initial Initial step size (same time units as the schedule).
  /// @param dt_min Minimum allowed adaptive step size.
  /// @param dt_max Maximum allowed adaptive step size.
  explicit dopri5(double rtol = default_rtol, double atol = default_atol,
                  double dt_initial = 0.01, double dt_min = 1e-6,
                  double dt_max = 1.0);

  /// @brief Integrate toward a specified time point using adaptive stepping.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration.
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current time and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator.
  std::shared_ptr<base_integrator> clone() override;

  /// @brief Adaptive-stepping statistics for diagnostics and tests.
  struct Stats {
    std::size_t accepted_steps = 0;
    std::size_t rejected_steps = 0;
    double min_dt_used = 1e300;
    double max_dt_used = 0.0;
    double avg_dt = 0.0;
  };
  /// @brief Return the adaptive-stepping statistics accrued so far.
  Stats getStats() const { return m_stats; }
  /// @brief Reset the adaptive-stepping statistics.
  void resetStats() { m_stats = Stats{}; }

private:
  double m_rtol;
  double m_atol;
  double m_dt;     // Current adaptive step size.
  double m_dt_min; // Minimum adaptive step size.
  double m_dt_max; // Maximum adaptive step size.
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  Stats m_stats;
};

/// @brief High-order commutator-free Magnus integrator (`magnus_cf4`).
///
/// A 4th-order, two-exponential commutator-free Magnus integrator specialised
/// for the (approximately) piecewise-constant Hamiltonian evolution that arises
/// in pulse schedules. For each step it materialises the dense Hamiltonian at
/// the two Gauss-Legendre nodes, forms the exact propagator via a GPU matrix
/// exponential (scaling-and-squaring Padé[13/13]), and reuses cached
/// propagators across identical time slices (an LRU propagator cache keyed by a
/// quantized Hamiltonian signature). Because the propagator is unitary, this
/// preserves norm/trace exactly and is efficient for repeated PWC segments.
///
/// This fast path applies only to *closed* systems evolved as density
/// matrices. For open systems (non-empty collapse operators / super-operator)
/// or state-vector evolution, the integrator transparently falls back to
/// `magnus_expansion` for correctness parity.
class magnus_cf4 : public cudaq::base_integrator {
public:
  /// @brief Default propagator-cache capacity (number of distinct slices).
  static constexpr std::size_t default_cache_capacity = 32;

  /// @brief Constructor.
  /// @param max_step_size Optional maximum internal sub-step size. When
  ///        provided the integrator sub-steps by at most this amount toward
  ///        each scheduled time point (recommended for time-dependent drives).
  /// @param cache_capacity Maximum number of cached propagators.
  explicit magnus_cf4(const std::optional<double> &max_step_size = {},
                      std::size_t cache_capacity = default_cache_capacity);

  ~magnus_cf4() override;

  /// @brief Integrate toward a specified time point.
  void integrate(double targetTime) override;
  /// @brief Set the initial state of the integration.
  void setState(const cudaq::state &initialState, double t0) override;
  /// @brief Get the current time and state.
  std::pair<double, cudaq::state> getState() override;
  /// @brief Clone the current integrator (fresh cache, same configuration).
  std::shared_ptr<base_integrator> clone() override;

  /// @brief Propagator-cache statistics for diagnostics and tests.
  struct Stats {
    std::size_t cache_hits = 0;
    std::size_t cache_misses = 0;
    std::size_t steps = 0;
    /// @brief Fraction of propagator lookups served from cache (0.0 - 1.0).
    double hit_rate() const {
      const std::size_t total = cache_hits + cache_misses;
      return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
    }
  };
  /// @brief Return the propagator-cache statistics accrued so far.
  Stats getStats() const { return m_stats; }
  /// @brief Reset the propagator-cache statistics.
  void resetStats() { m_stats = Stats{}; }

private:
  double m_t;
  std::shared_ptr<cudaq::state> m_state;
  std::optional<double> m_dt;
  std::size_t m_cache_capacity;
  Stats m_stats;

  // CUDA/cache resources are hidden behind a PImpl so this header stays free of
  // CUDA includes and is safe to include from CPU-only translation units.
  struct Impl;
  std::shared_ptr<Impl> m_impl;
};

} // namespace integrators
} // namespace cudaq
