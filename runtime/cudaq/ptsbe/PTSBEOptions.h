/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ShotAllocationStrategy.h"
#include <cstddef>
#include <memory>
#include <optional>

namespace cudaq::ptsbe {

// Forward declaration
class PTSSamplingStrategy;

/// @brief Configuration options for PTSBE execution.
///
/// Controls whether PTSBEExecutionData output is produced and which sampling
/// strategy to use.
///
/// The return_execution_data flag controls production of the full
/// PTSBEExecutionData, which bundles circuit instructions, trajectory
/// specifications, and per-trajectory measurement outcomes as a single unit.
///
struct PTSBEOptions {
  /// Produce PTSBEExecutionData (instructions + trajectories + measurement
  /// counts) in the sample result. Default false to avoid overhead when not
  /// needed.
  bool return_execution_data = false;

  /// Maximum number of unique trajectories to generate. When `nullopt`,
  /// defaults to the number of shots.
  std::optional<std::size_t> max_trajectories = std::nullopt;

  /// Custom sampling strategy. If `nullptr`, uses default strategy.
  std::shared_ptr<PTSSamplingStrategy> strategy = nullptr;

  /// Strategy for allocating shots across trajectories.
  /// Defaults to PROPORTIONAL.
  ShotAllocationStrategy shot_allocation{};
};

} // namespace cudaq::ptsbe
