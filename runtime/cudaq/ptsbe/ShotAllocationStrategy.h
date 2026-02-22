/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <ranges>
#include <span>

namespace cudaq::ptsbe {

/// @brief Strategy for allocating shots across selected trajectories
/// After trajectories are selected, this determines how many shots each
/// trajectory receives.
struct ShotAllocationStrategy {
  enum class Type {
    PROPORTIONAL,    // Shots proportional to trajectory weight
    UNIFORM,         // Equal shots per trajectory
    LOW_WEIGHT_BIAS, // Bias toward low-weight error trajectories
    HIGH_WEIGHT_BIAS // Bias toward high-weight error trajectories
  };

  Type type = Type::PROPORTIONAL;
  // Bias factor for weighted strategies (default: 2.0)
  double bias_strength = 2.0;
  // Random seed for multinomial sampling (PROPORTIONAL, biased strategies).
  // nullopt means use cudaq global seed if set, otherwise std::random_device.
  std::optional<std::uint64_t> seed = std::nullopt;

  /// @brief Default constructor
  ShotAllocationStrategy() = default;

  /// @brief Constructor with type
  /// @param t Allocation strategy type
  /// @param bias Bias strength for weighted strategies (default: 2.0)
  /// @param s Random seed for multinomial sampling (default: nullopt = auto)
  explicit ShotAllocationStrategy(Type t, double bias = 2.0,
                                  std::optional<std::uint64_t> s = std::nullopt)
      : type(t), bias_strength(bias), seed(s) {}
};

/// @brief Allocate shots across trajectories according to strategy
/// @param trajectories List of trajectories to allocate shots to
/// @param total_shots Total number of shots to distribute
/// @param strategy Shot allocation strategy
void allocateShots(std::span<cudaq::KrausTrajectory> trajectories,
                   std::size_t total_shots,
                   const ShotAllocationStrategy &strategy);

} // namespace cudaq::ptsbe
