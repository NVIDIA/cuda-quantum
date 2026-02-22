/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ConditionalSamplingStrategy.h"
#include <map>
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

  // Map from pattern to index in results for deduplication and multiplicity
  // accumulation. Patterns that failed the predicate are tracked separately.
  std::map<std::vector<std::size_t>, std::size_t> pattern_to_index;
  std::set<std::vector<std::size_t>> rejected_patterns;

  std::size_t total_possible = computeTotalTrajectories(noise_points);
  std::size_t actual_target = std::min(max_trajectories, total_possible);

  std::size_t trajectory_id = 0;
  std::size_t max_attempts = actual_target * ATTEMPT_MULTIPLIER;
  std::size_t attempts = 0;

  while (results.size() < max_trajectories && attempts < max_attempts) {
    attempts++;

    std::vector<KrausSelection> selections;
    std::vector<std::size_t> pattern;
    double probability = 1.0;

    selections.reserve(noise_points.size());
    pattern.reserve(noise_points.size());

    for (const auto &noise_point : noise_points) {
      std::discrete_distribution<std::size_t> dist(
          noise_point.channel.probabilities.begin(),
          noise_point.channel.probabilities.end());
      std::size_t sampled_idx = dist(rng_);
      pattern.push_back(sampled_idx);

      bool error = !noise_point.channel.is_identity_op(sampled_idx);
      selections.push_back(
          KrausSelection{noise_point.circuit_location, noise_point.qubits,
                         noise_point.op_name, sampled_idx, error});

      probability *= noise_point.channel.probabilities[sampled_idx];
    }

    auto it = pattern_to_index.find(pattern);
    if (it != pattern_to_index.end()) {
      results[it->second].multiplicity++;
    } else if (rejected_patterns.contains(pattern)) {
      // Already tested and failed predicate; skip.
    } else {
      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();

      if (predicate_(trajectory)) {
        pattern_to_index.emplace(std::move(pattern), results.size());
        results.push_back(std::move(trajectory));
        trajectory_id++;
      } else {
        rejected_patterns.insert(std::move(pattern));
      }
    }
  }

  for (auto &traj : results)
    traj.weight = static_cast<double>(traj.multiplicity);

  return results;
}

} // namespace cudaq::ptsbe
