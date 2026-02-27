/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/broadcast.h"
#include "../PTSSamplingStrategy.h"
#include <functional>
#include <random>

namespace cudaq::ptsbe {

/// @brief Predicate function type for filtering trajectories
/// @param trajectory The trajectory to evaluate
/// @return true if trajectory should be included, false otherwise
using TrajectoryPredicate = std::function<bool(const cudaq::KrausTrajectory &)>;

/// @brief Conditional trajectory sampling strategy
/// Samples trajectories that satisfy a user-defined predicate function.
class ConditionalSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with a predicate function and optional random seed
  /// @param predicate Function to filter trajectories
  /// @param seed Random seed for `reproducibility`. If 0 (default), uses
  /// CUDAQ's global random seed if set, otherwise std::random_device
  explicit ConditionalSamplingStrategy(TrajectoryPredicate predicate,
                                       std::uint64_t seed = 0)
      : predicate_(std::move(predicate)),
        rng_(seed == 0
                 ? (cudaq::get_random_seed() != 0 ? cudaq::get_random_seed()
                                                  : std::random_device{}())
                 : seed) {}

  /// @brief Destructor
  ~ConditionalSamplingStrategy() override;

  /// @brief Generate unique trajectories that satisfy the predicate
  ///
  /// Probabilistically samples trajectories from the noise model and filters
  /// them using the predicate function. Continues sampling until either
  /// `max_trajectories` matching trajectories are found or the attempt limit
  /// is reached.
  ///
  /// @param noise_points Noise information from circuit analysis. Each entry
  ///                     represents a location in the circuit where noise can
  ///                     be applied, with associated Kraus operators and
  ///                     probabilities.
  /// @param max_trajectories Maximum number of unique trajectories to generate.
  ///                         May return fewer if the predicate is very
  ///                         restrictive or if the attempt limit is reached.
  ///
  /// @return Vector of unique trajectories that pass the predicate filter.
  ///         Trajectories are ordered by sampling order (not by probability).
  ///
  /// Algorithm:
  /// 1. For each noise point, randomly sample a Kraus operator based on
  /// probabilities
  /// 2. Build a complete trajectory from all sampled operators
  /// 3. Check if trajectory pattern is unique
  /// 4. Apply predicate filter
  /// 5. If passed, add to results; otherwise, continue sampling
  /// 6. Stop when max_trajectories collected or max_attempts reached
  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const NoisePoint> noise_points,
                       std::size_t max_trajectories) const override;

  /// @brief Get strategy name
  /// @return "Conditional"
  [[nodiscard]] const char *name() const override { return "Conditional"; }

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy
  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<ConditionalSamplingStrategy>(*this);
  }

private:
  TrajectoryPredicate predicate_;
  mutable std::mt19937_64 rng_;
};

} // namespace cudaq::ptsbe
