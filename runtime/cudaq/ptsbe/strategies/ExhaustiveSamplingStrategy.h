/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "../PTSSamplingStrategy.h"

namespace cudaq::ptsbe {

/// @brief Exhaustive trajectory sampling strategy
/// Systematically enumerates all possible trajectories in lexicographic order.
class ExhaustiveSamplingStrategy : public PTSSamplingStrategy {
public:
  /// @brief Default constructor
  ExhaustiveSamplingStrategy() = default;

  /// @brief Destructor
  ~ExhaustiveSamplingStrategy() override;

  /// @brief Generate trajectories exhaustively in lexicographic order
  /// @param noise_points Noise information from circuit analysis
  /// @param max_trajectories Maximum number of unique trajectories to generate
  /// @return Vector of trajectories in lexicographic order (up to
  /// max_trajectories)
  [[nodiscard]] std::vector<cudaq::KrausTrajectory>
  generateTrajectories(std::span<const NoisePoint> noise_points,
                       std::size_t max_trajectories) const override;

  /// @brief Get strategy name
  /// @return "Exhaustive"
  [[nodiscard]] const char *name() const override { return "Exhaustive"; }

  /// @brief Clone this strategy
  /// @return Unique pointer to a copy
  [[nodiscard]] std::unique_ptr<PTSSamplingStrategy> clone() const override {
    return std::make_unique<ExhaustiveSamplingStrategy>(*this);
  }
};

/// @brief Enumerate trajectories in lexicographic order over operator indices.
///
/// When index_mapping is non-empty, index_mapping[i] remaps enumeration
/// positions to actual operator indices for noise point i (e.g. sorted by
/// descending probability). When empty, raw indices 0..N are used.
std::vector<cudaq::KrausTrajectory>
enumerateLexicographic(std::span<const NoisePoint> noise_points,
                       std::size_t limit,
                       std::span<const std::vector<std::size_t>> index_mapping);

} // namespace cudaq::ptsbe
