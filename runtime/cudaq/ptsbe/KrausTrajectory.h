/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausSelection.h"
#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

inline constexpr double PROBABILITY_EPSILON = 1e-9;

/// @brief Represents one complete path through the space of possible noise
/// realizations
struct KrausTrajectory {
  /// @brief Unique identifier for this trajectory
  std::size_t trajectory_id = 0;

  /// @brief Complete specification of which Kraus operators to apply at each
  /// noise point This tracks only the injected noise operators
  std::vector<KrausSelection> kraus_selections;

  /// @brief Computed probability of this trajectory occurring
  /// This is the product of individual Kraus operator probabilities
  double probability = 0.0;

  /// @brief Number of measurement shots allocated to this trajectory
  std::size_t num_shots = 0;

  /// @brief The measurement results for this specific trajectory
  std::optional<std::map<std::string, std::size_t>> measurement_counts;

  /// @brief Default constructor
  KrausTrajectory() = default;

  /// @brief Constructor for trajectory creation
  /// @param id Unique identifier for this trajectory
  /// @param selections Complete specification of Kraus operators at each noise
  /// point
  /// @param prob Computed probability of this trajectory occurring
  /// @param shots Number of measurement shots allocated to this trajectory
  KrausTrajectory(std::size_t id, std::vector<KrausSelection> selections,
                  double prob, std::size_t shots)
      : trajectory_id(id), kraus_selections(std::move(selections)),
        probability(prob), num_shots(shots) {}

  /// @brief Equality comparison for testing
  /// @param other KrausTrajectory to compare with
  /// @return true if trajectory_id, selections, probability, and num_shots
  /// match
  constexpr bool operator==(const KrausTrajectory &other) const {
    return trajectory_id == other.trajectory_id &&
           kraus_selections == other.kraus_selections &&
           std::abs(probability - other.probability) < PROBABILITY_EPSILON &&
           num_shots == other.num_shots;
  }
};

} // namespace cudaq
