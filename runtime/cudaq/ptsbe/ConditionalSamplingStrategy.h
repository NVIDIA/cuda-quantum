/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "PTSSamplingStrategy.h"
#include <functional>

namespace cudaq::ptsbe {

/// @brief Predicate function type for filtering trajectories
/// @param trajectory The trajectory to evaluate
/// @return true if trajectory should be included, false otherwise
using TrajectoryPredicate = std::function<bool(const cudaq::KrausTrajectory &)>;

/// @brief Conditional trajectory sampling strategy
/// Samples trajectories that satisfy a user-defined predicate function.
class ConditionalSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Construct with a predicate function
  /// @param predicate Function to filter trajectories
  explicit ConditionalSamplingStrategy(TrajectoryPredicate predicate)
      : predicate_(std::move(predicate)) {}

  /// @brief Destructor
  ~ConditionalSamplingStrategy() override;

  /// @brief Generate unique trajectories that satisfy the predicate
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories to generate
  /// @return Vector of unique trajectories that pass the predicate filter
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
};

} // namespace cudaq::ptsbe
