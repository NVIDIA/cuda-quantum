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
/// Currently supports Gate, Noise, and Measurement for static circuits.
///
// NOTE: For mid-circuit measurement support, a Branch instruction type
// could be added. A Branch would contain child PTSBETrace instances (one
// per measurement outcome), each with its own instructions and
// trajectories. This forms a tree with the top-level tracing gates
// before the first measurement, then branches into sub-traces that
// continue independently with different quantum states. Roughly the
// same would apply for classical control-flow tracing.
enum class TraceInstructionType {
  Gate,       /// Quantum gate operation (H, X, CNOT, RX, etc.)
  Noise,      /// Noise channel location (depolarizing, amplitude_damping, etc.)
  Measurement /// Terminal measurement operation
};

/// @brief Single operation in the PTSBE execution trace.
///
/// Stores gate, noise channel, or measurement info with plain qubit indices.
/// This is the user-facing trace type exposed to Python via pybind11.
///
struct TraceInstruction {
  /// @brief Instruction category (Gate, Noise, or Measurement)
  TraceInstructionType type;

  /// @brief Operation name (e.g., `h`, `depolarizing`, `mz`)
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
/// One trace may have many trajectories which reference the noise locations
/// within the trace.
struct PTSBETrace {
  /// @brief Ordered circuit operations (gates, noise channels, measurements)
  std::vector<TraceInstruction> instructions;

  /// @brief The sampled trajectories
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Count instructions matching the given type and optional name
  std::size_t
  count_instructions(TraceInstructionType type,
                     std::optional<std::string> name = std::nullopt) const;

  /// @brief Look up a trajectory by its ID
  /// @return Reference to the trajectory if found, std::nullopt otherwise
  std::optional<std::reference_wrapper<const cudaq::KrausTrajectory>>
  get_trajectory(std::size_t trajectoryId) const;
};

} // namespace cudaq::ptsbe
