/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file PTSBESamplerImpl.h
/// @brief Internal header for PTSBE simulator integration.
///
/// This header exposes template functions and types that depend on
/// `nvqir::CircuitSimulator` internals. It is intended for simulator
/// implementations and tests, not for the public API. The public header
/// PTSBESampler.h provides the stable API (PTSBatch, PTSBECapable, etc.)
/// without leaking `nvqir` internals.

#pragma once

#include "KrausTrajectory.h"
#include "PTSBEExecutionData.h"
#include "PTSBESampler.h"
#include "common/Trace.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <cstddef>
#include <span>
#include <vector>

namespace cudaq::ptsbe {

// Abstract interface for batch simulator
// Simulators can optionally implement this interface to provide a custom
// implementation of sampleWithPTSBE.
struct BatchSimulator {
  virtual ~BatchSimulator() = default;
  virtual std::vector<cudaq::sample_result>
  sampleWithPTSBE(const PTSBatch &batch) = 0;
};

/// @brief Alias for CircuitSimulator gate task type
template <typename ScalarType>
using GateTask =
    typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;

/// @brief Convert a PTSBE TraceInstruction (Gate type) to a simulator task.
/// Looks up the gate matrix from the registry and maps plain qubit IDs.
template <typename ScalarType>
GateTask<ScalarType> convertToSimulatorTask(const TraceInstruction &inst);

/// @brief Convert a PTSBE trace to a simulator task list, keeping only Gate
/// entries (Noise and Measurement entries are skipped).
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
convertTrace(std::span<const TraceInstruction> ptsbeTrace);

/// @brief Convert a KrausSelection to a GateApplicationTask using the
/// noise channel's unitary operators from the trace instruction.
template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const TraceInstruction &noiseInst);

/// @brief Walk the PTSBE trace and build the merged task list for one
/// trajectory. Gate entries become gate tasks, Noise entries are resolved
/// via the trajectory selections (channel looked up from the trace), and
/// Measurement entries are skipped (terminal measurements are handled
/// separately by the simulator).
///
/// @param includeIdentity When true, identity Kraus operators are
///   included as gate tasks. Useful if you require all trajectories to have
///   identical gate structure.
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity = false);

/// @brief Generic PTSBE execution implementation
///
/// For each trajectory:
/// - Resets simulator to computational zero state
/// - Merges PTSBE trace with trajectory noise selections
/// - Applies merged gate tasks
/// - Samples measurement qubits
///
/// Returns per-trajectory results for flexibility. Use aggregateResults()
/// to combine into a single sample_result if needed.
///
/// This is the fallback implementation used when a simulator does not
/// provide a custom sampleWithPTSBE() method.
///
/// Caller must set up ExecutionContext and allocate qubits before
/// calling this function. Caller is also responsible for de-allocating qubits
/// and resetting the ExecutionContext after this function returns.
///
/// @tparam ScalarType Simulator scalar type
/// @param simulator Circuit simulator instance (must have ExecutionContext set)
/// @param batch PTSBE specification
/// @return Per-trajectory sample results
/// @throws std::runtime_error if ExecutionContext not set or gate conversion
/// fails
template <typename ScalarType>
std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
                   const PTSBatch &batch);

} // namespace cudaq::ptsbe
