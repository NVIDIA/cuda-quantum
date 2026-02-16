/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ProbabilisticSamplingStrategy.h"
#include <set>

namespace cudaq::ptsbe {

/// @brief Multiplier for maximum sampling attempts relative to target
/// trajectories.
static constexpr std::size_t ATTEMPT_MULTIPLIER = 10;

ProbabilisticSamplingStrategy::~ProbabilisticSamplingStrategy() = default;

std::vector<cudaq::KrausTrajectory>
ProbabilisticSamplingStrategy::generateTrajectories(
    std::span<const NoisePoint> noise_points,
    std::size_t max_trajectories) const {

  std::vector<cudaq::KrausTrajectory> results;
  results.reserve(max_trajectories);

  if (noise_points.empty()) {
    return results;
  }

  std::set<std::vector<std::size_t>> seen_patterns;

  // Total possible unique trajectories with overflow protection
  // If we want more trajectories than exist, cap at total to avoid infinite
  // sampling
  std::size_t total_possible = computeTotalTrajectories(noise_points);
  std::size_t actual_target = std::min(max_trajectories, total_possible);

  std::size_t trajectory_id = 0;
  const std::size_t base_attempts = actual_target * ATTEMPT_MULTIPLIER;
  constexpr std::size_t MAX_ATTEMPTS_CAP = 500000;
  std::size_t min_attempts_for_coverage =
      (total_possible <= 10000)
          ? std::min(total_possible * 5000, MAX_ATTEMPTS_CAP)
          : base_attempts;
  std::size_t max_attempts = std::max(base_attempts, min_attempts_for_coverage);
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

      selections.push_back(KrausSelection{
          noise_point.circuit_location, noise_point.qubits, noise_point.op_name,
          static_cast<KrausOperatorType>(sampled_idx)});

      probability *= noise_point.channel.probabilities[sampled_idx];
    }

    if (seen_patterns.insert(pattern).second) {
      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id++)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();
      results.push_back(std::move(trajectory));

      // If we've found all possible unique trajectories, stop
      if (results.size() >= total_possible) {
        break;
      }
    }
  }

  return results;
}

} // namespace cudaq::ptsbe
