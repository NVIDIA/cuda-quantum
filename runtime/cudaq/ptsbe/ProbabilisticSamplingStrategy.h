/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSSamplingStrategy.h"
#include <random>

namespace cudaq::ptsbe {

/// @brief Probabilistic trajectory sampling strategy
/// Samples trajectories randomly based on their occurrence probabilities.
class ProbabilisticSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with optional random seed
  /// @param seed Random seed for `reproducibility`
  explicit ProbabilisticSamplingStrategy(
      std::uint64_t seed = std::random_device{}())
      : rng_(seed) {}

  /// @brief Destructor
  ~ProbabilisticSamplingStrategy() override;

  /// @brief Generate unique trajectories using probability-weighted random
  /// selection
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of UNIQUE trajectories to generate
  /// @return Vector of unique randomly sampled trajectories (no duplicates)
  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const NoisePoint> noise_points,
                       std::size_t max_trajectories) const override;

  /// @brief Get strategy name
  /// @return "Probabilistic"
  [[nodiscard]] const char *name() const override { return "Probabilistic"; }

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy
  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<ProbabilisticSamplingStrategy>(*this);
  }

private:
  mutable std::mt19937_64 rng_;
};

} // namespace cudaq::ptsbe
