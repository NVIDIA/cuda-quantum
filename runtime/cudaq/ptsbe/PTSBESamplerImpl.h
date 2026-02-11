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
#include "PTSSamplingStrategy.h"
#include "common/Trace.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <cstddef>
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

/// @brief Convert Trace instruction to simulator task
template <typename ScalarType>
GateTask<ScalarType>
convertToSimulatorTask(const cudaq::Trace::Instruction &inst);

/// @brief Convert entire kernel trace to simulator task list
template <typename ScalarType>
std::vector<GateTask<ScalarType>> convertTrace(const cudaq::Trace &trace);

/// @brief Convert a KrausSelection to a GateApplicationTask using the
/// noise channel's unitary operators for the matrix
template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const NoisePoint &noiseSite);

/// @brief Merge base tasks with trajectory noise insertions
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(const std::vector<GateTask<ScalarType>> &baseTasks,
                         const cudaq::KrausTrajectory &trajectory,
                         const std::vector<NoisePoint> &noiseSites);

/// @brief Merge kernel trace with trajectory noise to produce task list
template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeAndConvert(const cudaq::Trace &kernelTrace,
                const cudaq::KrausTrajectory &trajectory,
                const std::vector<NoisePoint> &noiseSites);

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
