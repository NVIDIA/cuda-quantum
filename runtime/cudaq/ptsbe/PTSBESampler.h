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
#include "nvqir/Gates.h"
#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cudaq::ptsbe {

/// @brief Batch specification for PTSBE execution
struct PTSBatch {
  /// @brief Captured kernel circuit
  cudaq::Trace kernelTrace;

  /// @brief Sampled noise trajectories
  std::vector<cudaq::KrausTrajectory> trajectories;

  /// @brief Qubits to measure (terminal measurements)
  /// NOTE: This currently only applies to kernels that are terminal measurement
  /// only which is a limitation of the current PTSBE implementation.
  std::vector<std::size_t> measureQubits;

  /// @brief Calculate total shots across all trajectories
  std::size_t totalShots() const {
    std::size_t total = 0;
    for (const auto &traj : trajectories)
      total += traj.num_shots;
    return total;
  }
};

/// @brief Concept for simulators supporting a customized PTSBE implementation
///
/// Enables compile-time detection of simulator PTSBE support with zero runtime
/// overhead. Simulators opting into PTSBE should implement sampleWithPTSBE
/// returning per-trajectory results. We use a concept to avoid exposing
/// the simulator base class to PTSBE.
template <typename SimulatorType>
concept PTSBECapable = requires(SimulatorType &sim, const PTSBatch &batch) {
  {
    sim.sampleWithPTSBE(batch)
  } -> std::same_as<std::vector<cudaq::sample_result>>;
};

/// @brief Alias for CircuitSimulator gate task type
template <typename ScalarType>
using GateTask =
    typename nvqir::CircuitSimulatorBase<ScalarType>::GateApplicationTask;

/// @brief Convert Trace instruction to simulator task
///
/// Looks up gate matrix from nvqir::Gates.h registry and constructs
/// a GateApplicationTask with typed parameters and qubit indices.
///
/// @tparam ScalarType Simulator scalar type (float or double)
/// @param inst Trace instruction containing gate name, parameters, controls,
/// targets
/// @return GateApplicationTask with computed unitary matrix
/// @throws std::runtime_error if gate name is not recognized
template <typename ScalarType>
GateTask<ScalarType>
convertToSimulatorTask(const cudaq::Trace::Instruction &inst) {
  // Convert parameters to ScalarType
  std::vector<ScalarType> typedParams;
  typedParams.reserve(inst.params.size());
  for (auto p : inst.params)
    typedParams.push_back(static_cast<ScalarType>(p));

  // Look up gate matrix from registry (throws for unknown gates)
  auto gateName = nvqir::getGateNameFromString(inst.name);
  auto matrix = nvqir::getGateByName<ScalarType>(gateName, typedParams);

  // Extract qubit IDs from QuditInfo
  std::vector<std::size_t> controls;
  controls.reserve(inst.controls.size());
  for (const auto &q : inst.controls)
    controls.push_back(q.id);

  std::vector<std::size_t> targets;
  targets.reserve(inst.targets.size());
  for (const auto &q : inst.targets)
    targets.push_back(q.id);

  return GateTask<ScalarType>(inst.name, matrix, controls, targets, typedParams);
}

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
std::vector<GateTask<ScalarType>> convertTrace(const cudaq::Trace &trace) {
  std::vector<GateTask<ScalarType>> tasks;
  tasks.reserve(trace.getNumInstructions());
  for (const auto &inst : trace)
    tasks.push_back(convertToSimulatorTask<ScalarType>(inst));
  return tasks;
}

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
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel) {
  std::string gateName =
      (sel.kraus_operator_index == KrausOperatorType::IDENTITY) ? "id"
                                                                : sel.op_name;
  auto gateEnum = nvqir::getGateNameFromString(gateName);
  auto matrix = nvqir::getGateByName<ScalarType>(gateEnum, {});
  return GateTask<ScalarType>(gateName, matrix, {}, sel.qubits, {});
}

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
                         const cudaq::KrausTrajectory &trajectory) {
  const auto &selections = trajectory.kraus_selections;

  std::vector<GateTask<ScalarType>> merged;
  merged.reserve(baseTasks.size() + selections.size());

  std::size_t noiseIdx = 0;
  for (std::size_t gateIdx = 0; gateIdx < baseTasks.size(); ++gateIdx) {
    merged.push_back(baseTasks[gateIdx]);

    // Insert all noise for this gate location
    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == gateIdx) {
      merged.push_back(krausSelectionToTask<ScalarType>(selections[noiseIdx]));
      ++noiseIdx;
    }
  }

  // Validate: any remaining noise has invalid circuit_location
  if (noiseIdx < selections.size()) {
    throw std::runtime_error(
        "Invalid circuit_location: " +
        std::to_string(selections[noiseIdx].circuit_location) +
        " >= " + std::to_string(baseTasks.size()));
  }

  return merged;
}

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
                const cudaq::KrausTrajectory &trajectory) {
  auto baseTasks = convertTrace<ScalarType>(kernelTrace);
  return mergeTasksWithTrajectory<ScalarType>(baseTasks, trajectory);
}

/// @brief Aggregate per-trajectory sample results into a single result
///
/// Combines counts from all trajectory results into one sample_result.
/// This is useful for the final aggregation step after PTSBE execution.
///
/// @param results Vector of per-trajectory sample results
/// @return Single aggregated sample_result
cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results);

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
/// calling this function. Caller is also responsible for deallocating qubits
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

/// @brief Execute PTSBE batch on current simulator
///
/// Handles both runtime precision dispatch and compile-time concept dispatch:
/// 1. Uses isSinglePrecision() to determine float vs double
/// 2. Checks PTSBECapable concept for custom simulator implementations
/// 3. Falls back to samplePTSBEGeneric if no custom implementation
///
/// Caller must have set up ExecutionContext and allocated qubits
/// on the simulator before calling this function.
///
/// @param batch PTSBatch with kernelTrace, trajectories, and measureQubits
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or contract violated
std::vector<cudaq::sample_result> samplePTSBE(const PTSBatch &batch);

/// @brief Execute PTSBE with full lifecycle management (registry-based)
///
/// Convenience function that handles the complete simulator lifecycle:
/// 1. Gets current simulator from registry
/// 2. Creates ExecutionContext with specified type
/// 3. Sets context on simulator and allocates qubits
/// 4. Calls samplePTSBE for precision dispatch and trajectory execution
/// 5. Deallocates qubits and resets context
///
/// @param batch PTSBE specification
/// @param contextType ExecutionContext type (default: "sample")
/// @return Per-trajectory sample results
/// @throws std::runtime_error if simulator cast fails or gate conversion fails
std::vector<cudaq::sample_result>
samplePTSBEWithLifecycle(const PTSBatch &batch,
                          const std::string &contextType = "sample");

} // namespace cudaq::ptsbe
