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

static constexpr std::size_t ATTEMPT_MULTIPLIER = 10;

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

  // MC draw budget. Either user-specified or auto-calculated.
  // The loop stops once max_trajectories unique patterns are found,
  // so this is a ceiling on draws. Every draw contributes to
  // exactly one trajectory's multiplicity which provides an
  // unbiased MC estimator: p_k ≈ weight_k / M_total.
  constexpr std::size_t MAX_BUDGET_CAP = 500000;
  std::size_t budget;
  if (trajectory_samples_.has_value()) {
    budget = std::max(max_trajectories, trajectory_samples_.value());
  } else {
    std::size_t target = std::min(max_trajectories, total_possible);
    budget = std::min(target * ATTEMPT_MULTIPLIER, MAX_BUDGET_CAP);
    budget = std::max(max_trajectories, budget);
  }

  for (std::size_t sample = 0; sample < budget; ++sample) {
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
    } else {
      pattern_to_index.emplace(std::move(pattern), results.size());
      auto trajectory = KrausTrajectory::builder()
                            .setId(trajectory_id++)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build();
      results.push_back(std::move(trajectory));
      if (results.size() >= max_trajectories)
        break;
    }
  }

  for (auto &traj : results)
    traj.weight = static_cast<double>(traj.multiplicity);

  return results;
}

} // namespace cudaq::ptsbe
