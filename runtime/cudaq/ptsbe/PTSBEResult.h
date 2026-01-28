/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include <cstddef>
#include <vector>

// Forward declaration for GateApplicationTask
namespace nvqir {
template <typename ScalarType>
struct GateApplicationTask;
}

namespace cudaq::ptsbe {

/// @brief Complete result from PreTrajectorySamplingEngine
template <typename ScalarType>
struct PTSBEResult {
  /// @brief Task lists - one complete trajectory per inner vector
  /// Each inner vector contains GateApplicationTasks (original gates + noise operations)
  std::vector<std::vector<nvqir::GateApplicationTask<ScalarType>>> task_lists;

  /// @brief Trajectory metadata
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Qubits to measure (same for all trajectories)
  std::vector<std::size_t> measure_qubits;

  /// @brief Default constructor
  PTSBEResult() = default;

  /// @brief Check if result is valid
  /// @return true if task_lists and trajectories have matching sizes and are non-empty
  [[nodiscard]] constexpr bool isValid() const {
    return task_lists.size() == trajectories.size() &&
           !task_lists.empty();
  }

  /// @brief Get number of trajectories
  /// @return Number of trajectories in this result
  [[nodiscard]] constexpr std::size_t numTrajectories() const {
    return task_lists.size();
  }
};

} // namespace cudaq::ptsbe
