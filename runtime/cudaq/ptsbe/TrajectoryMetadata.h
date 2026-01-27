/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausSelection.h"
#include "KrausTrajectory.h"
#include <cstddef>
#include <vector>
#include <cmath>

namespace cudaq::ptsbe {

/// @brief Metadata for a single trajectory
struct TrajectoryMetadata {
    /// @brief Unique identifier for this trajectory
    std::size_t trajectory_id = 0;
    
    /// @brief Which errors occurred in this trajectory
    std::vector<cudaq::KrausSelection> kraus_selections;
    
    /// @brief Probability of this trajectory
    double probability = 0.0;
    
    /// @brief Number of shots allocated
    std::size_t num_shots = 0;
    
    /// @brief Default constructor
    TrajectoryMetadata() = default;
    
    /// @brief Constructor with all fields
    /// @param id Unique identifier for this trajectory
    /// @param sels Which errors occurred in this trajectory
    /// @param prob Probability of this trajectory
    /// @param shots Number of shots allocated
    TrajectoryMetadata(std::size_t id, 
                      std::vector<cudaq::KrausSelection> sels,
                      double prob,
                      std::size_t shots)
        : trajectory_id(id),
          kraus_selections(std::move(sels)),
          probability(prob),
          num_shots(shots) {}
    
    /// @brief Constructor from KrausTrajectory
    /// @param traj Source KrausTrajectory to extract metadata from
    explicit TrajectoryMetadata(const cudaq::KrausTrajectory& traj)
        : trajectory_id(traj.trajectory_id),
          kraus_selections(traj.kraus_selections),
          probability(traj.probability),
          num_shots(traj.num_shots) {}
    
    /// @brief Equality comparison for testing
    /// @param other TrajectoryMetadata to compare with
    /// @return true if all fields match
    constexpr bool operator==(const TrajectoryMetadata& other) const {
        return trajectory_id == other.trajectory_id &&
               kraus_selections == other.kraus_selections &&
               std::abs(probability - other.probability) < 1e-9 &&
               num_shots == other.num_shots;
    }
};

} // namespace cudaq::ptsbe
