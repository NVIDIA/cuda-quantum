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

  std::map<std::vector<std::size_t>, std::size_t> pattern_to_index;
  std::set<std::vector<std::size_t>> rejected_patterns;

  std::size_t total_possible = computeTotalTrajectories(noise_points);
  std::size_t actual_target = std::min(max_trajectories, total_possible);

  std::size_t trajectory_id = 0;
  std::size_t max_attempts = actual_target * ATTEMPT_MULTIPLIER;
  std::size_t attempts = 0;

  std::vector<std::discrete_distribution<std::size_t>> distributions;
  distributions.reserve(noise_points.size());
  for (const auto &np : noise_points)
    distributions.emplace_back(np.channel.probabilities.begin(),
                               np.channel.probabilities.end());

  std::vector<std::size_t> pattern(noise_points.size());

  while (results.size() < max_trajectories && attempts < max_attempts) {
    attempts++;

    double probability = 1.0;
    for (std::size_t i = 0; i < noise_points.size(); ++i) {
      std::size_t idx = distributions[i](rng_);
      pattern[i] = idx;
      probability *= noise_points[i].channel.probabilities[idx];
    }

    auto it = pattern_to_index.find(pattern);
    if (it != pattern_to_index.end()) {
      results[it->second].multiplicity++;
    } else if (!rejected_patterns.contains(pattern)) {
      std::vector<KrausSelection> selections;
      selections.reserve(noise_points.size());
      for (std::size_t i = 0; i < noise_points.size(); ++i) {
        const auto &np = noise_points[i];
        bool error = !np.channel.is_identity_op(pattern[i]);
        selections.push_back(KrausSelection{np.circuit_location, np.qubits,
                                            np.op_name, pattern[i], error});
      }

      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();

      if (predicate_(trajectory)) {
        pattern_to_index.emplace(pattern, results.size());
        results.push_back(std::move(trajectory));
        trajectory_id++;
      } else {
        rejected_patterns.insert(pattern);
      }
    }
  }

  for (auto &traj : results)
    traj.weight = static_cast<double>(traj.multiplicity);

  return results;
}

} // namespace cudaq::ptsbe
