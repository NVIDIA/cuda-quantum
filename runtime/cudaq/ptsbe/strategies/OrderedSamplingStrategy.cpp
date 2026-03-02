/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrderedSamplingStrategy.h"
#include "ExhaustiveSamplingStrategy.h"
#include <algorithm>
#include <numeric>

namespace cudaq::ptsbe {

static constexpr std::size_t GENERATION_MULTIPLIER = 10;

OrderedSamplingStrategy::~OrderedSamplingStrategy() = default;

std::vector<cudaq::KrausTrajectory>
OrderedSamplingStrategy::generateTrajectories(
    std::span<const detail::NoisePoint> noise_points,
    std::size_t max_trajectories) const {

  if (noise_points.empty())
    return {};

  std::size_t total = detail::computeTotalTrajectories(noise_points);

  // Sort operator indices by descending probability so the lexicographic
  // prefix contains the highest-probability trajectories first.
  std::vector<std::vector<std::size_t>> sorted_indices(noise_points.size());
  for (std::size_t i = 0; i < noise_points.size(); ++i) {
    auto &si = sorted_indices[i];
    si.resize(noise_points[i].channel.size());
    std::iota(si.begin(), si.end(), 0);
    std::ranges::sort(si, [&](auto a, auto b) {
      return noise_points[i].channel.probabilities[a] >
             noise_points[i].channel.probabilities[b];
    });
  }

  std::size_t generation_limit =
      std::min(total, max_trajectories * GENERATION_MULTIPLIER);

  auto results = detail::enumerateLexicographic(noise_points, generation_limit,
                                                sorted_indices);

  std::ranges::sort(results, [](const auto &a, const auto &b) {
    return a.probability > b.probability;
  });

  if (results.size() > max_trajectories)
    results.resize(max_trajectories);

  for (std::size_t i = 0; i < results.size(); ++i)
    results[i].trajectory_id = i;

  return results;
}

} // namespace cudaq::ptsbe
