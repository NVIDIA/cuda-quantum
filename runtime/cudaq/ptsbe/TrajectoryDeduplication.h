/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.  *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include <cstddef>
#include <span>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Hash the content of a trajectory (kraus_selections) for deduplication
///
/// Two trajectories that have the same kraus_selections (order and values)
/// produce the same hash. trajectory_id, num_shots, and multiplicity are
/// ignored.
///
/// @param trajectory Trajectory to hash
/// @return Hash value suitable for use in hash-based merge
[[nodiscard]] std::size_t
hashTrajectoryContent(const cudaq::KrausTrajectory &trajectory);

/// @brief Deduplicate trajectories by content and track multiplicity
///
/// Trajectories with identical kraus_selections are merged into a single
/// representative.
///
/// @param trajectories Input list of trajectories (may contain duplicates)
/// @return Deduplicated list; each trajectory has multiplicity >= 1
[[nodiscard]] std::vector<cudaq::KrausTrajectory>
deduplicateTrajectories(std::span<const cudaq::KrausTrajectory> trajectories);

} // namespace cudaq::ptsbe
