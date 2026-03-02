/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include <cstddef>
#include <span>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Hash the content (kraus_selections) of a trajectory.
/// @param trajectory KrausTrajectory to hash
/// @return Hash value; equal content yields equal hash
std::size_t hashTrajectoryContent(const cudaq::KrausTrajectory &trajectory);

/// @brief `Deduplicate` trajectories by content (kraus_selections).
/// Representatives keep the first occurrence's probability, trajectory_id, and
/// num_shots; multiplicity is the sum of all merged trajectories'
/// multiplicities
/// @param trajectories Input trajectories (order preserved for representative
/// choice)
/// @return Unique trajectories with multiplicity set
std::vector<cudaq::KrausTrajectory>
deduplicateTrajectories(std::span<const cudaq::KrausTrajectory> trajectories);

} // namespace cudaq::ptsbe
