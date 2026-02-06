/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "common/NoiseModel.h"
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Discriminator for instruction types within the PTSBE trace.
///
/// Phase 1 supports Gate, Noise, and Measurement. Phase 2 will add Branch
/// for mid-circuit measurement support.
enum class TraceInstructionType {
  Gate,  ///< Quantum gate operation (H, X, CNOT, RX, etc.)
  Noise, ///< Noise channel location (depolarizing, amplitude_damping, etc.)
  Measurement ///< Terminal measurement operation
};

/// @brief Single operation in the PTSBE execution trace.
///
/// Stores gate, noise channel, or measurement info with plain qubit indices.
/// This is the user-facing trace type exposed to Python via pybind11.
///
/// Gate example: {Gate, "h", {0}, {}, {}}
/// Noise example: {Noise, "depolarizing", {0}, {}, {0.01}}
/// Measurement example: {Measurement, "mz", {0}, {}, {}}
struct TraceInstruction {
  /// @brief Instruction category (Gate, Noise, or Measurement)
  TraceInstructionType type;

  /// @brief Operation name (e.g., "h", "depolarizing", "mz")
  std::string name;

  /// @brief Target qubit indices
  std::vector<std::size_t> targets;

  /// @brief Control qubit indices (empty for non-controlled operations)
  std::vector<std::size_t> controls;

  /// @brief Parameters (gate angles or noise channel parameters)
  std::vector<double> params;

  /// @brief Noise channel (populated only for Noise instructions)
  std::optional<cudaq::kraus_channel> channel;

  /// @brief Default constructor
  TraceInstruction() = default;

  /// @brief Constructor with all fields
  TraceInstruction(TraceInstructionType type, std::string name,
                   std::vector<std::size_t> targets,
                   std::vector<std::size_t> controls,
                   std::vector<double> params,
                   std::optional<cudaq::kraus_channel> channel = std::nullopt)
      : type(type), name(std::move(name)), targets(std::move(targets)),
        controls(std::move(controls)), params(std::move(params)),
        channel(std::move(channel)) {}
};

/// @brief Container for PTSBE trace data including circuit structure and
/// trajectory data.
///
/// The trace represents the circuit structure (what operations were applied
/// and where noise channels exist), while trajectories represent noise
/// realizations (which Kraus operators were selected).
///
/// One trace, many trajectories: PTSBE samples noise upfront, creating
/// multiple trajectories that share the same circuit trace. Trajectories
/// with identical Kraus selections are merged (num_shots increases).
struct PTSBETrace {
  /// @brief Ordered circuit operations (gates, noise channels, measurements)
  std::vector<TraceInstruction> instructions;

  /// @brief All trajectory specifications with their outcomes
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Look up a trajectory by its ID
  /// @param trajectoryId The trajectory ID to look up
  /// @return Reference to the trajectory if found, std::nullopt otherwise
  std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>>
  get_trajectory(std::size_t trajectoryId) const;
};

} // namespace cudaq::ptsbe
