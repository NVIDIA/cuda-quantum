/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "TrajectoryMetadata.h"
#include <cstddef>
#include <vector>

// Forward declaration for GateApplicationTask
namespace nvqir {
template <typename ScalarType>
struct GateApplicationTask;
}

namespace cudaq::ptsbe {

/// @brief Complete result from PreTrajectorySamplingEngine
/// Contains task lists for execution and metadata for annotation
template <typename ScalarType>
struct PTSBEResult {
  /// @brief Task lists - one complete trajectory per inner vector
  /// Each GateApplicationTask is either a gate or a deterministic noise
  /// operation (Kraus op) Task lists contain only unitary operations - noise
  /// channels are pre-sampled
  std::vector<std::vector<nvqir::GateApplicationTask<ScalarType>>> task_lists;

  /// @brief Metadata for each trajectory
  /// Index i in metadata corresponds to task_lists[i]
  std::vector<TrajectoryMetadata> metadata;

  /// @brief Shots allocated to each trajectory
  std::vector<std::size_t> shots_per_trajectory;

  /// @brief Qubits to measure
  std::vector<std::size_t> measure_qubits;

  /// @brief Default constructor
  PTSBEResult() = default;

  /// @brief Check if result is valid
  /// @return true if all vectors have matching sizes and are non-empty
  [[nodiscard]] constexpr bool isValid() const {
    return task_lists.size() == metadata.size() &&
           task_lists.size() == shots_per_trajectory.size() &&
           !task_lists.empty();
  }

  /// @brief Get number of trajectories
  /// @return Number of trajectories in this result
  [[nodiscard]] constexpr std::size_t numTrajectories() const {
    return task_lists.size();
  }
};

} // namespace cudaq::ptsbe
