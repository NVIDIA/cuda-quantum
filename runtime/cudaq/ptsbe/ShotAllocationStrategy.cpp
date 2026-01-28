/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ShotAllocationStrategy.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

void allocateShots(std::span<cudaq::KrausTrajectory> trajectories,
                   std::size_t total_shots,
                   const ShotAllocationStrategy &strategy) {

  if (trajectories.empty()) {
    throw std::invalid_argument(
        "Cannot allocate shots to empty trajectory list");
  }

  if (total_shots == 0) {
    // Zero shots requested
    for (auto &traj : trajectories) {
      traj.num_shots = 0;
    }
    return;
  }

  switch (strategy.type) {
  case ShotAllocationStrategy::Type::PROPORTIONAL: {
    // Allocate shots proportional to trajectory probability
    double total_probability = 0.0;
    for (const auto &traj : trajectories) {
      total_probability += traj.probability;
    }

    if (total_probability <= 0.0) {
      throw std::invalid_argument(
          "Total probability must be positive for proportional allocation");
    }

    for (auto &traj : trajectories) {
      traj.num_shots = static_cast<std::size_t>(
          (traj.probability / total_probability) * total_shots);
    }
    break;
  }

  case ShotAllocationStrategy::Type::UNIFORM: {
    // Equal shots per trajectory
    std::size_t shots_per_traj = total_shots / trajectories.size();
    std::size_t remainder = total_shots % trajectories.size();

    for (std::size_t i = 0; i < trajectories.size(); ++i) {
      // Base allocation to all, plus 1 extra to first 'remainder' trajectories
      trajectories[i].num_shots = shots_per_traj + (i < remainder ? 1 : 0);
    }
    break;
  }

  case ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS: {
    // Bias toward trajectories with fewer errors
    std::vector<double> weights;
    weights.reserve(trajectories.size());

    for (const auto &traj : trajectories) {
      std::size_t error_count = traj.countErrors();
      // weight = (1 + error_count)^(-bias_strength) * probability
      // Lower error_count → higher weight
      double weight = std::pow(1.0 + error_count, -strategy.bias_strength) *
                      traj.probability;
      weights.push_back(weight);
    }

    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_weight <= 0.0) {
      throw std::invalid_argument(
          "Total weight must be positive for biased allocation");
    }

    for (std::size_t i = 0; i < trajectories.size(); ++i) {
      trajectories[i].num_shots =
          static_cast<std::size_t>((weights[i] / total_weight) * total_shots);
    }
    break;
  }

  case ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS: {
    // Bias toward trajectories with more errors
    std::vector<double> weights;
    weights.reserve(trajectories.size());

    for (const auto &traj : trajectories) {
      std::size_t error_count = traj.countErrors();
      // weight = (1 + error_count)^(+bias_strength) * probability
      // Higher error_count → higher weight
      double weight = std::pow(1.0 + error_count, strategy.bias_strength) *
                      traj.probability;
      weights.push_back(weight);
    }

    double total_weight = std::accumulate(weights.begin(), weights.end(), 0.0);
    if (total_weight <= 0.0) {
      throw std::invalid_argument(
          "Total weight must be positive for biased allocation");
    }

    for (std::size_t i = 0; i < trajectories.size(); ++i) {
      trajectories[i].num_shots =
          static_cast<std::size_t>((weights[i] / total_weight) * total_shots);
    }
    break;
  }
  }

  // Handle rounding errors
  std::size_t allocated = 0;
  for (const auto &traj : trajectories) {
    allocated += traj.num_shots;
  }

  if (allocated < total_shots && !trajectories.empty()) {
    // Distribute remaining shots evenly among trajectories
    std::size_t remaining = total_shots - allocated;
    std::size_t num_trajectories = trajectories.size();
    std::size_t shots_per_traj =
        (remaining + num_trajectories - 1) / num_trajectories;

    for (auto &traj : trajectories) {
      if (remaining == 0)
        break;

      std::size_t shots_to_add = std::min(shots_per_traj, remaining);
      traj.num_shots += shots_to_add;
      remaining -= shots_to_add;
    }
  } else if (allocated > total_shots && !trajectories.empty()) {
    // Remove excess shots, distributing the removals
    std::size_t excess = allocated - total_shots;
    std::size_t num_trajectories = trajectories.size();
    std::size_t shots_per_traj =
        (excess + num_trajectories - 1) / num_trajectories;

    for (auto &traj : trajectories) {
      if (excess == 0)
        break;

      std::size_t shots_to_remove =
          std::min({shots_per_traj, excess, traj.num_shots});
      traj.num_shots -= shots_to_remove;
      excess -= shots_to_remove;
    }
  }
}

} // namespace cudaq::ptsbe
