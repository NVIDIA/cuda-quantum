/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudm_helpers.h>
#include "cudm_state.h>
#include "runtime/common/EvolveResult.h"
#include <complex>
#include <cudaq/base_integrator.h>
#include <cudaq/helpers.h>
#include <cudaq/operators.h>
#include <cudaq/schedule.h>
#include <stdexcept>
#include <vector>

namespace cudaq {

// Configuration struct for the solver
struct Config {
  std::map<int, int> dimensions;                // Hilbert space dimensions
  operator_sum hamiltonian;                     // Hamiltonian operator
  std::vector<operator_sum> collapse_operators; // Collapse operators
  std::vector<operator_sum> observables;        // Observables to evaluate
  std::variant<InitialState, std::vector<std::complex<double>>>
      initial_state;                       // Initial state
  Schedule schedule;                       // Evolution schedule
  bool store_intermediate_results = false; // Flag to store intermediate states
};

class cudm_solver {
public:
  cudm_solver(const Config &config);

  void validate_config();

  cudm_state initialize_state();

  void evolve(cudm_state &state, cudensitymatOperator_t &liouvillian,
              const std::vector<cudensitymatOperator_t> &obervable_ops,
              evolve_result &result);

  evolve_result evolve_dynamics();

  cudensitymatOperator_t construct_liouvillian(
      cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
      const std::vector<cudensitymatOperator_t> &collapse_operators,
      bool me_solve);

private:
  Config config_;
};
} // namespace cudaq