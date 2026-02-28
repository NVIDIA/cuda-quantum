/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExhaustiveSamplingStrategy.h"
#include <algorithm>

std::vector<cudaq::KrausTrajectory> cudaq::ptsbe::enumerateLexicographic(
    std::span<const NoisePoint> noise_points, std::size_t limit,
    std::span<const std::vector<std::size_t>> index_mapping) {
  std::vector<cudaq::KrausTrajectory> results;
  if (noise_points.empty())
    return results;

  const bool has_mapping = !index_mapping.empty();

  std::vector<std::size_t> operator_counts;
  operator_counts.reserve(noise_points.size());
  for (const auto &np : noise_points)
    operator_counts.push_back(np.channel.size());

  results.reserve(limit);
  std::vector<std::size_t> indices(noise_points.size(), 0);

  for (std::size_t trajectory_id = 0; trajectory_id < limit; ++trajectory_id) {
    std::vector<KrausSelection> selections;
    selections.reserve(noise_points.size());
    double probability = 1.0;

    for (std::size_t i = 0; i < noise_points.size(); ++i) {
      const auto &np = noise_points[i];
      std::size_t op_idx =
          has_mapping ? index_mapping[i][indices[i]] : indices[i];

      bool error = !np.channel.is_identity_op(op_idx);
      selections.push_back(KrausSelection{np.circuit_location, np.qubits,
                                          np.op_name, op_idx, error});
      probability *= np.channel.probabilities[op_idx];
    }

    results.push_back(KrausTrajectory::builder()
                          .setId(trajectory_id)
                          .setSelections(std::move(selections))
                          .setProbability(probability)
                          .build());

    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i]++;
      if (indices[i] < operator_counts[i])
        break;
      indices[i] = 0;
    }
  }

  return results;
}

cudaq::ptsbe::ExhaustiveSamplingStrategy::~ExhaustiveSamplingStrategy() =
    default;

std::vector<cudaq::KrausTrajectory>
cudaq::ptsbe::ExhaustiveSamplingStrategy::generateTrajectories(
    std::span<const NoisePoint> noise_points,
    std::size_t max_trajectories) const {
  std::size_t total = computeTotalTrajectories(noise_points);
  return enumerateLexicographic(noise_points, std::min(total, max_trajectories),
                                {});
}
