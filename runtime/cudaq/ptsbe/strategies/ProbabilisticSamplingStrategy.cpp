/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ProbabilisticSamplingStrategy.h"
#include <map>

namespace cudaq::ptsbe {

ProbabilisticSamplingStrategy::~ProbabilisticSamplingStrategy() = default;

std::vector<cudaq::KrausTrajectory>
ProbabilisticSamplingStrategy::generateTrajectories(
    std::span<const NoisePoint> noise_points,
    std::size_t max_trajectories) const {

  std::vector<cudaq::KrausTrajectory> results;

  if (noise_points.empty() || max_trajectories == 0)
    return results;

  std::size_t total_possible = computeTotalTrajectories(noise_points);
  results.reserve(std::min(max_trajectories, total_possible));

  // Map from Kraus-index pattern to position in results, used both for
  // deduplication and for tracking how many times each pattern was drawn
  // (multiplicity).
  std::map<std::vector<std::size_t>, std::size_t> pattern_to_index;
  std::size_t trajectory_id = 0;

  // Draw max_trajectories total Monte Carlo samples. Each sample either
  // discovers a new unique trajectory or increments an existing trajectory's
  // multiplicity. With enough samples the multiplicities converge to the true
  // probability distribution, which is required for correct weighting during
  // shot allocation and result aggregation.
  for (std::size_t sample = 0; sample < max_trajectories; ++sample) {
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

    auto [it, inserted] = pattern_to_index.emplace(pattern, results.size());
    if (inserted) {
      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id++)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();
      results.push_back(std::move(trajectory));
    } else {
      results[it->second].multiplicity++;
    }
  }

  return results;
}

} // namespace cudaq::ptsbe
