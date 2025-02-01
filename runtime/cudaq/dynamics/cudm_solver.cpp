/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/cudm_solver.h"
#include "cudaq/base_time_stepper.h"
#include "cudaq/cudm_helpers.h"
#include "cudaq/cudm_state.h"

namespace cudaq {
cudm_solver::cudm_solver(const Config &config) : config_(config) {
  validate_config();
}

void cudm_solver::validate_config() {
  if (config_.dimensions.empty()) {
    throw std::invalid_argument("Dimensions map cannot be empty.");
  }

  if (config_.hamiltonian.get_terms().empty()) {
    throw std::invalid_argument("Hamiltonian must have at least one term.");
  }

  if (config_.dimensions.empty()) {
    throw std::invalid_argument("Schedule cannot be empty.");
  }
}

cudm_state cudm_solver::initialize_state() {
  std::vector<int64_t> mode_extents;
  for (const auto &[key, value] : config_.dimensions) {
    mode_extents.push_back(value);
  }

  return cudm_state::create_initial_state(config_.initial_state, mode_extents,
                                          !config_.collapse_operators.empty());
}

cudensitymatOperator_t cudm_solver::construct_liouvillian(
    cudensitymatHandle_t handle, const cudensitymatOperator_t &hamiltonian,
    const std::vector<cudensitymatOperator_t> &collapse_operators,
    bool me_solve) {
  return construct_liovillian(handle, hamiltonian, collapse_operators,
                              me_solve ? 1.0 : 0.0);
}

void cudm_solver::evolve(
    cudm_state &state, cudensitymatOperator_t &liouvillian,
    const std::vector<cudensitymatOperator_t> &observable_ops,
    evolve_result &result) {
  auto handle = state.get_impl();

  // Initialize the stepper
  BaseTimeStepper timeStepper(liouvillian, handle);
}
} // namespace cudaq