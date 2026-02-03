/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ConditionalSamplingStrategy.h"
#include <ranges>

namespace cudaq::ptsbe {

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

  std::size_t total_trajectories = 1;
  std::vector<std::size_t> operator_counts;
  operator_counts.reserve(noise_points.size());

  for (const auto &noise_point : noise_points) {
    std::size_t count = noise_point.kraus_operators.size();
    operator_counts.push_back(count);
    total_trajectories *= count;
  }

  std::vector<std::size_t> indices(noise_points.size(), 0);
  std::size_t trajectory_id = 0;

  for (std::size_t candidate_id = 0;
       candidate_id < total_trajectories && results.size() < max_trajectories;
       ++candidate_id) {

    std::vector<KrausSelection> selections;
    selections.reserve(noise_points.size());
    double probability = 1.0;

    for (std::size_t i = 0; i < noise_points.size(); ++i) {
      const auto &noise_point = noise_points[i];
      std::size_t op_idx = indices[i];

      selections.push_back(KrausSelection{
          noise_point.circuit_location, noise_point.qubits, noise_point.op_name,
          static_cast<KrausOperatorType>(op_idx)});

      probability *= noise_point.probabilities[op_idx];
    }

    auto trajectory = KrausTrajectory::builder()
                          .setId(trajectory_id)
                          .setSelections(std::move(selections))
                          .setProbability(probability)
                          .build();

    if (predicate_(trajectory)) {
      results.push_back(std::move(trajectory));
      trajectory_id++;
    }

    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i]++;
      if (indices[i] < operator_counts[i]) {
        break;
      }
      indices[i] = 0;
    }
  }

  return results;
}

} // namespace cudaq::ptsbe
