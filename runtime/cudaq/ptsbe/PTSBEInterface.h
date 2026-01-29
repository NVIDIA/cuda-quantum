/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "KrausTrajectory.h"
#include "common/Trace.h"
#include "nvqir/CircuitSimulator.h"
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Batch specification for PTSBE execution
struct PTSBatch {
  /// @brief Captured kernel circuit
  cudaq::Trace kernel_trace;

  /// @brief Sampled noise trajectories
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Qubits to measure (terminal measurements)
  /// NOTE: This currently only applies to kernels that are terminal measurement only
  /// which is a limitation of the current PTSBE implementation.
  std::vector<std::size_t> measure_qubits;
};

/// @brief Concept for simulators supporting a customized PTSBE implementation
///
/// Enables compile-time detection of simulator PTSBE support with zero runtime
/// overhead. Simulators opting into PTSBE should implement sampleWithPTSBE
/// returning per-trajectory results. We use a concept to avoid exposing
/// the simulator base class to PTSBE.
template <typename SimulatorType>
concept PTSBECapable =
    requires(SimulatorType &sim, const PTSBatch &batch) {
      {
        sim.sampleWithPTSBE(batch)
      } -> std::same_as<std::vector<cudaq::sample_result>>;
    };

/// @brief Execute PTSBE batch with compile-time dispatch
///
/// @param simulator Circuit simulator instance
/// @param batch Batch specification
/// @return Aggregated execution result
/// @throws std::runtime_error Not yet implemented
template <typename ScalarType>
cudaq::sample_result
executePTSBE(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
             const PTSBatch &batch) {
  throw std::runtime_error("executePTSBE: Not implemented");
}

/// @brief Convert Trace instruction to simulator task
///
/// @param inst Trace instruction
/// @return Simulator task with matrix and parameters
/// @throws std::runtime_error Not yet implemented
template <typename ScalarType>
nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask
convertToSimulatorTask(const cudaq::Trace::Instruction &inst) {
  throw std::runtime_error("convertToSimulatorTask: Not implemented");
}

/// @brief Merge kernel trace with trajectory noise to produce task list to execute on simulator
///
/// @param kernel_trace Base kernel circuit
/// @param trajectory Sampled trajectory with noise
/// @return Complete task list for simulator
/// @throws std::runtime_error Not yet implemented
template <typename ScalarType>
std::vector<typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask>
mergeAndConvert(const cudaq::Trace &kernel_trace,
                const cudaq::KrausTrajectory &trajectory) {
  throw std::runtime_error("mergeAndConvert: Not implemented");
}

} // namespace cudaq::ptsbe
