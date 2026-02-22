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
#include <optional>
#include <random>

namespace cudaq::ptsbe {

/// @brief Probabilistic trajectory sampling strategy
/// Samples trajectories randomly based on their occurrence probabilities.
class ProbabilisticSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with optional random seed and trajectory sample count
  /// @param seed Random seed for reproducibility. When nullopt (default), uses
  /// CUDAQ's global random seed if set, otherwise std::random_device.
  /// @param trajectory_samples Total number of Monte Carlo trajectory samples
  /// to draw. Controls the accuracy/cost tradeoff. More samples discover
  /// rarer error trajectories and produce more accurate weights for shot
  /// allocation. When nullopt (default) is provided, a budget is calculated
  //  using a small multiplier of max_trajectories. For low-error-rate circuits,
  /// increase this value to discover higher-order error trajectories.
  explicit ProbabilisticSamplingStrategy(
      std::optional<std::uint64_t> seed = std::nullopt,
      std::optional<std::size_t> trajectory_samples = std::nullopt)
      : rng_(seed.value_or(cudaq::get_random_seed() != 0
                               ? cudaq::get_random_seed()
                               : std::random_device{}())),
        trajectory_samples_(trajectory_samples) {}

  /// @brief Destructor
  ~ProbabilisticSamplingStrategy() override;

  /// @brief Generate trajectories via probability-weighted Monte Carlo sampling
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories requested.
  /// The actual number of Monte Carlo samples drawn is
  /// max(max_trajectories, trajectory_samples), where trajectory_samples is
  /// either the user-specified value or an auto-calculated exploration budget.
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
  std::optional<std::size_t> trajectory_samples_;
};

} // namespace cudaq::ptsbe
