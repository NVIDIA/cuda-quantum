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

  std::map<std::vector<std::size_t>, std::size_t> pattern_to_index;
  std::size_t trajectory_id = 0;

  std::vector<std::discrete_distribution<std::size_t>> distributions;
  distributions.reserve(noise_points.size());
  for (const auto &np : noise_points)
    distributions.emplace_back(np.channel.probabilities.begin(),
                               np.channel.probabilities.end());

  // MC draw budget. Either user-specified or auto-calculated.
  // The loop stops once max_trajectories unique patterns are found,
  // so this is a ceiling on draws.
  std::size_t budget;
  if (max_trajectory_samples_.has_value()) {
    budget = std::max(max_trajectories, max_trajectory_samples_.value());
  } else {
    std::size_t target = std::min(max_trajectories, total_possible);
    budget = std::min(target * ATTEMPT_MULTIPLIER, MAX_BUDGET_CAP);
    budget = std::max(max_trajectories, budget);
  }

  std::vector<std::size_t> pattern(noise_points.size());

  for (std::size_t sample = 0; sample < budget; ++sample) {
    double probability = 1.0;
    for (std::size_t i = 0; i < noise_points.size(); ++i) {
      std::size_t idx = distributions[i](rng_);
      pattern[i] = idx;
      probability *= noise_points[i].channel.probabilities[idx];
    }

    auto it = pattern_to_index.find(pattern);
    if (it != pattern_to_index.end()) {
      results[it->second].multiplicity++;
    } else {
      std::vector<KrausSelection> selections;
      selections.reserve(noise_points.size());
      for (std::size_t i = 0; i < noise_points.size(); ++i) {
        const auto &np = noise_points[i];
        bool error = !np.channel.is_identity_op(pattern[i]);
        selections.push_back(KrausSelection{np.circuit_location, np.qubits,
                                            np.op_name, pattern[i], error});
      }

      pattern_to_index.emplace(pattern, results.size());
      results.push_back(KrausTrajectory::builder()
                            .setId(trajectory_id++)
                            .setSelections(std::move(selections))
                            .setProbability(probability)
                            .build());
      if (results.size() >= max_trajectories)
        break;
    }
  }

  for (auto &traj : results)
    traj.weight = static_cast<double>(traj.multiplicity);

  return results;
}

} // namespace cudaq::ptsbe
