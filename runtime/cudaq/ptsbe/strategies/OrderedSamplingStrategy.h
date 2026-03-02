/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../PTSSamplingStrategy.h"
#include <algorithm>
#include <ranges>

namespace cudaq::ptsbe {

/// @brief Ordered trajectory sampling strategy
/// Samples trajectories sorted by probability in descending order.
class OrderedSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Default constructor
  OrderedSamplingStrategy() = default;

  /// @brief Destructor
  ~OrderedSamplingStrategy() override;

  /// @brief Generate top-k trajectories sorted by probability (descending)
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories to generate
  /// @return Vector of top-k highest-probability trajectories
  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const detail::NoisePoint> noise_points,
                       std::size_t max_trajectories) const override;

  /// @brief Get strategy name
  /// @return "Ordered"
  [[nodiscard]] const char *name() const override { return "Ordered"; }

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy
  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<OrderedSamplingStrategy>(*this);
  }
};

} // namespace cudaq::ptsbe
