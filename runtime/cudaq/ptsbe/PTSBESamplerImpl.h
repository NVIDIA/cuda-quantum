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
#include "PTSBESampler.h"
#include "common/Trace.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <cstddef>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Alias for CircuitSimulator gate task type
template <typename ScalarType>
using GateTask =
    typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;

/// @brief Convert Trace instruction to simulator task
///
/// Looks up gate matrix from `nvqir::Gates.h` registry and constructs
/// a GateApplicationTask with typed parameters and qubit indices.
///
/// @tparam ScalarType Simulator scalar type (float or double)
/// @param inst Trace instruction containing gate name, parameters, controls,
/// targets
/// @return GateApplicationTask with computed unitary matrix
/// @throws std::runtime_error if gate name is not recognized
template <typename ScalarType>
GateTask<ScalarType>
convertToSimulatorTask(const cudaq::Trace::Instruction &inst);

/// @brief Convert entire kernel trace to simulator task list
///
/// Applies convertToSimulatorTask to each instruction in the trace.
/// Result can be reused across multiple trajectory merges.
///
/// @tparam ScalarType Simulator scalar type (float or double)
/// @param trace Kernel trace with gate instructions
/// @return Vector of GateApplicationTask ready for simulator execution
/// @throws std::runtime_error if any instruction has unrecognized gate
template <typename ScalarType>
std::vector<GateTask<ScalarType>> convertTrace(const cudaq::Trace &trace);

/// @brief Convert a KrausSelection to a GateApplicationTask
///
/// @tparam ScalarType Simulator scalar type
/// @param sel KrausSelection specifying the noise operation
/// @return GateApplicationTask ready for simulator execution
///
/// TODO: Currently uses op_name string lookup as a workaround. When
/// KrausOperatorType is expanded to include named error types (X_ERROR,
/// Y_ERROR, Z_ERROR, etc.), this should map directly from enum to gate.
template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel);

/// @brief Merge base tasks with trajectory noise insertions
///
/// Performs merge of base circuit tasks with noise
/// operations specified in trajectory. Noise is inserted after
/// the gate at circuit_location.
///
/// @tparam ScalarType Simulator scalar type
/// @param baseTasks Pre-converted base circuit tasks
/// @param trajectory Trajectory with noise selections
/// @return Merged task list ready for execution
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(const std::vector<GateTask<ScalarType>> &baseTasks,
                         const cudaq::KrausTrajectory &trajectory);

/// @brief Merge kernel trace with trajectory noise to produce task list to
/// execute on simulator
///
/// Convenience function that converts trace to tasks, then merges with noise.
///
/// @param kernelTrace Base kernel circuit
/// @param trajectory Sampled trajectory with noise
/// @return Complete task list for simulator
/// @throws std::runtime_error if gate name not recognized or invalid location
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeAndConvert(const cudaq::Trace &kernelTrace,
                const cudaq::KrausTrajectory &trajectory);

/// @brief Generic PTSBE execution implementation
///
/// Converts base trace once, then for each trajectory:
/// - Resets simulator to computational zero state
/// - Applies noise merged circuit
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
