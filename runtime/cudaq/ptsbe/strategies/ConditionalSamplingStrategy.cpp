/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ConditionalSamplingStrategy.h"
#include <set>

namespace cudaq::ptsbe {

/// @brief Multiplier for maximum sampling attempts relative to target
/// trajectories.
static constexpr std::size_t ATTEMPT_MULTIPLIER = 100;

ConditionalSamplingStrategy::~ConditionalSamplingStrategy() = default;

std::vector<cudaq::KrausTrajectory>
ConditionalSamplingStrategy::generateTrajectories(
    std::span<const NoisePoint> noise_points,
    std::size_t max_trajectories) const {

  std::vector<cudaq::KrausTrajectory> results;
  results.reserve(max_trajectories);

  if (noise_points.empty()) {
    return results;
  }

  // Track unique trajectory patterns to avoid duplicates
  // Pattern = sequence of Kraus operator indices [op0_idx, op1_idx, ...]
  std::set<std::vector<std::size_t>> seen_patterns;

  std::size_t total_possible = computeTotalTrajectories(noise_points);
  std::size_t actual_target = std::min(max_trajectories, total_possible);

  std::size_t trajectory_id = 0;
  std::size_t max_attempts = actual_target * ATTEMPT_MULTIPLIER;
  std::size_t attempts = 0;

  // Sample until we have max_trajectories unique trajectories that pass the
  // predicate
  while (results.size() < max_trajectories && attempts < max_attempts) {
    attempts++;

    // For each noise point (location where noise can occur):
    // - Sample which Kraus operator to apply at that location
    // - indices[i] selects from noise_points[i].kraus_operators
    std::vector<KrausSelection> selections;
    std::vector<std::size_t> pattern;
    double probability = 1.0;

    selections.reserve(noise_points.size());
    pattern.reserve(noise_points.size());

    // Sample trajectory: for each noise point, choose which Kraus operator to
    // apply
    for (const auto &noise_point : noise_points) {
      // Use discrete distribution to sample according to operator probabilities
      std::discrete_distribution<std::size_t> dist(
          noise_point.channel.probabilities.begin(),
          noise_point.channel.probabilities.end());
      std::size_t sampled_idx = dist(rng_);
      pattern.push_back(sampled_idx);

      // Build KrausSelection: "at circuit_location, apply Kraus operator
      // #sampled_idx". Conversion to simulator task happens later in
      // krausSelectionToTask() in PTSBESampler.cpp.
      bool error = !noise_point.channel.is_identity_op(sampled_idx);
      selections.push_back(
          KrausSelection{noise_point.circuit_location, noise_point.qubits,
                         noise_point.op_name, sampled_idx, error});

      probability *= noise_point.channel.probabilities[sampled_idx];
    }

    if (seen_patterns.insert(pattern).second) {
      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();

      if (predicate_(trajectory)) {
        results.push_back(std::move(trajectory));
        trajectory_id++;
      }
      // If predicate fails, trajectory is discarded and we continue sampling
    }
    // If duplicate pattern, discard and continue sampling
  }

  return results;
}

} // namespace cudaq::ptsbe
