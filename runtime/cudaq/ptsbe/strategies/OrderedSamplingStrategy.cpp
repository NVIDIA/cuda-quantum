/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrderedSamplingStrategy.h"

namespace cudaq::ptsbe {

/// @brief Multiplier for generation limit before sorting.
static constexpr std::size_t GENERATION_MULTIPLIER = 10;

OrderedSamplingStrategy::~OrderedSamplingStrategy() = default;

std::vector<cudaq::KrausTrajectory>
OrderedSamplingStrategy::generateTrajectories(
    std::span<const NoisePoint> noise_points,
    std::size_t max_trajectories) const {

  std::vector<cudaq::KrausTrajectory> results;

  if (noise_points.empty()) {
    return results;
  }

  std::size_t total_trajectories = computeTotalTrajectories(noise_points);

  std::vector<std::size_t> operator_counts;
  operator_counts.reserve(noise_points.size());
  for (const auto &noise_point : noise_points) {
    operator_counts.push_back(noise_point.channel.size());
  }

  std::size_t generation_limit =
      std::min(total_trajectories, max_trajectories * GENERATION_MULTIPLIER);
  results.reserve(generation_limit);

  std::vector<std::size_t> indices(noise_points.size(), 0);

  for (std::size_t trajectory_id = 0; trajectory_id < generation_limit;
       ++trajectory_id) {
    std::vector<KrausSelection> selections;
    selections.reserve(noise_points.size());
    double probability = 1.0;

    for (std::size_t i = 0; i < noise_points.size(); ++i) {
      const auto &noise_point = noise_points[i];
      std::size_t op_idx = indices[i];

      bool error = !noise_point.channel.is_identity_op(op_idx);
      selections.push_back(KrausSelection{noise_point.circuit_location,
                                          noise_point.qubits,
                                          noise_point.op_name, op_idx, error});

      probability *= noise_point.channel.probabilities[op_idx];
    }

    results.push_back(KrausTrajectory::builder()
                          .setId(trajectory_id)
                          .setSelections(std::move(selections))
                          .setProbability(probability)
                          .build());

    for (std::size_t i = 0; i < indices.size(); ++i) {
      indices[i]++;
      if (indices[i] < operator_counts[i]) {
        break;
      }
      indices[i] = 0;
    }
  }

  std::ranges::sort(results, [](const auto &a, const auto &b) {
    return a.probability > b.probability;
  });

  if (results.size() > max_trajectories) {
    results.resize(max_trajectories);
  }

  for (std::size_t i = 0; i < results.size(); ++i) {
    results[i].trajectory_id = i;
  }

  return results;
}

} // namespace cudaq::ptsbe
