/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/broadcast.h"
#include "../PTSSamplingStrategy.h"
#include <random>

namespace cudaq::ptsbe {

/// @brief Probabilistic trajectory sampling strategy
/// Samples trajectories randomly based on their occurrence probabilities.
class ProbabilisticSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with optional random seed
  /// @param seed Random seed for `reproducibility`. If 0 (default), uses
  /// CUDAQ's global random seed if set, otherwise std::random_device
  explicit ProbabilisticSamplingStrategy(std::uint64_t seed = 0)
      : rng_(seed == 0
                 ? (cudaq::get_random_seed() != 0 ? cudaq::get_random_seed()
                                                  : std::random_device{}())
                 : seed) {}

  /// @brief Destructor
  ~ProbabilisticSamplingStrategy() override;

  /// @brief Generate trajectories via probability-weighted Monte Carlo sampling
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Total number of trajectory samples to draw. Each
  /// sample either discovers a new unique trajectory or increments an existing
  /// one's multiplicity, so that multiplicities converge to the true
  /// probability distribution.
  /// @return Vector of unique trajectories with accumulated multiplicities
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
