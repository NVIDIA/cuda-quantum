/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "base_time_stepper.h"
#include "operators.h"
#include "schedule.h"
#include <map>
#include <memory>
#include <vector>

namespace cudaq {
template <typename TState, typename HandlerTy = std::complex<double>>
class BaseIntegrator {
protected:
  std::map<std::string, double> integrator_options;
  TState state;
  double t;
  std::map<int, int> dimensions;
  std::shared_ptr<Schedule> schedule;
  std::shared_ptr<operator_sum<HandlerTy>> hamiltonian;
  std::shared_ptr<BaseTimeStepper<TState>> stepper;
  std::vector<std::shared_ptr<operator_sum<HandlerTy>>> collapse_operators;

  virtual void post_init() = 0;

public:
  /// @brief Default constructor
  BaseIntegrator() = default;

  /// @brief Constructor to initialize the integrator with a state and time
  /// stepper.
  /// @param initial_state Initial quantum state.
  /// @param t0 Initial time.
  /// @param stepper Time stepper instance.
  BaseIntegrator(TState &&initial_state, double t0,
                 std::shared_ptr<BaseTimeStepper<TState>> stepper)
      : state(std::move(initial_state)), t(t0), stepper(std::move(stepper)) {
    if (!this->stepper) {
      throw std::runtime_error("Time stepper is not initialized.");
    }
  }

  virtual ~BaseIntegrator() = default;

  /// @brief Set the initial state and time
  void set_state(const TState &initial_state, double t0 = 0.0) {
    state = initial_state;
    t = t0;
  }

  /// @brief Set an option for the integrator
  void set_option(const std::string &key, double value) {
    integrator_options[key] = value;
  }

  /// @brief Set the system parameters (dimensions, schedule, and operators)
  void set_system(const std::map<int, int> &dimensions,
                  std::shared_ptr<Schedule> schedule,
                  std::shared_ptr<operator_sum<HandlerTy>> hamiltonian,
                  std::vector<std::shared_ptr<operator_sum<HandlerTy>>>
                      collapse_operators = {}) {
    this->dimensions = dimensions;
    this->schedule = schedule;
    this->hamiltonian = hamiltonian;
    this->collapse_operators = collapse_operators;
  }

  /// @brief Perform integration to the target time.
  virtual void integrate(double target_time) = 0;

  /// @brief Get the current time and state.
  std::pair<double, const TState &> get_state() const { return {t, state}; }
};
} // namespace cudaq
