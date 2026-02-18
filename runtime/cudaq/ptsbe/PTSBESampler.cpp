/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESamplerImpl.h"
#include "cudaq/simulators.h"
#include <numeric>
#include <stdexcept>

namespace cudaq::ptsbe {

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

  return GateTask<ScalarType>(inst.name, matrix, controls, targets,
                              typedParams);
}

template <typename ScalarType>
std::vector<GateTask<ScalarType>> convertTrace(const cudaq::Trace &trace) {
  std::vector<GateTask<ScalarType>> tasks;
  tasks.reserve(trace.getNumInstructions());
  for (const auto &inst : trace) {
    // Skip apply_noise; they become noise insertions only, not gate tasks
    if (inst.type == cudaq::TraceInstructionType::Noise)
      continue;
    tasks.push_back(convertToSimulatorTask<ScalarType>(inst));
  }
  return tasks;
}

template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const NoisePoint &noiseSite) {
  auto k = static_cast<std::size_t>(sel.kraus_operator_index);
  const auto &unitaryDouble = noiseSite.channel.unitary_ops.at(k);
  std::vector<std::complex<ScalarType>> matrix;
  matrix.reserve(unitaryDouble.size());
  for (const auto &elem : unitaryDouble)
    matrix.emplace_back(static_cast<ScalarType>(elem.real()),
                        static_cast<ScalarType>(elem.imag()));
  std::string opName;
  if (k < noiseSite.channel.op_names.size())
    opName = noiseSite.channel.op_names[k];
  else
    opName = noiseSite.channel.get_type_name() + "[" + std::to_string(k) + "]";
  return GateTask<ScalarType>(opName, matrix, {}, sel.qubits, {});
}

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(const std::vector<GateTask<ScalarType>> &baseTasks,
                         const cudaq::KrausTrajectory &trajectory,
                         const std::vector<NoisePoint> &noiseSites) {
  const auto &selections = trajectory.kraus_selections;

  std::vector<GateTask<ScalarType>> merged;
  merged.reserve(baseTasks.size() + selections.size());

  std::size_t noiseIdx = 0;
  for (std::size_t gateIdx = 0; gateIdx < baseTasks.size(); ++gateIdx) {
    merged.push_back(baseTasks[gateIdx]);

    // Insert all noise for this gate location
    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == gateIdx) {
      merged.push_back(krausSelectionToTask<ScalarType>(selections[noiseIdx],
                                                        noiseSites[noiseIdx]));
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

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeAndConvert(const cudaq::Trace &kernelTrace,
                const cudaq::KrausTrajectory &trajectory,
                const std::vector<NoisePoint> &noiseSites) {
  auto baseTasks = convertTrace<ScalarType>(kernelTrace);
  return mergeTasksWithTrajectory<ScalarType>(baseTasks, trajectory,
                                              noiseSites);
}

// ---------------------------------------------------------------------------
// Explicit template instantiations for float and double
// ---------------------------------------------------------------------------

template GateTask<float>
convertToSimulatorTask<float>(const cudaq::Trace::Instruction &);
template GateTask<double>
convertToSimulatorTask<double>(const cudaq::Trace::Instruction &);

template std::vector<GateTask<float>> convertTrace<float>(const cudaq::Trace &);
template std::vector<GateTask<double>>
convertTrace<double>(const cudaq::Trace &);

template GateTask<float>
krausSelectionToTask<float>(const cudaq::KrausSelection &, const NoisePoint &);
template GateTask<double>
krausSelectionToTask<double>(const cudaq::KrausSelection &, const NoisePoint &);

template std::vector<GateTask<float>>
mergeTasksWithTrajectory<float>(const std::vector<GateTask<float>> &,
                                const cudaq::KrausTrajectory &,
                                const std::vector<NoisePoint> &);
template std::vector<GateTask<double>>
mergeTasksWithTrajectory<double>(const std::vector<GateTask<double>> &,
                                 const cudaq::KrausTrajectory &,
                                 const std::vector<NoisePoint> &);

template std::vector<GateTask<float>>
mergeAndConvert<float>(const cudaq::Trace &, const cudaq::KrausTrajectory &,
                       const std::vector<NoisePoint> &);
template std::vector<GateTask<double>>
mergeAndConvert<double>(const cudaq::Trace &, const cudaq::KrausTrajectory &,
                        const std::vector<NoisePoint> &);

// ---------------------------------------------------------------------------
// Non-template implementations
// ---------------------------------------------------------------------------

cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results) {
  if (results.empty())
    return cudaq::sample_result{};

  cudaq::CountsDictionary aggregatedCounts;
  for (const auto &res : results) {
    for (const auto &[bitstring, count] : res.to_map())
      aggregatedCounts[bitstring] += count;
  }
  return cudaq::sample_result{cudaq::ExecutionResult{aggregatedCounts}};
}

template <typename ScalarType>
std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
                   const PTSBatch &batch) {
  if (!cudaq::getExecutionContext())
    throw std::runtime_error(
        "samplePTSBEGeneric requires ExecutionContext to be set. "
        "Use cudaq::detail::setExecutionContext() before invoking.");

  if (batch.trajectories.empty())
    return {};

  std::size_t totalShots = batch.totalShots();
  if (totalShots == 0)
    return {};

  if (batch.measureQubits.empty())
    return {};

  auto baseTasks = convertTrace<ScalarType>(batch.kernelTrace);

  std::vector<cudaq::sample_result> results;
  results.reserve(batch.trajectories.size());

  for (const auto &traj : batch.trajectories) {
    if (traj.num_shots == 0) {
      // Push empty result to maintain index correspondence with trajectories
      results.push_back(cudaq::sample_result{
          cudaq::ExecutionResult{cudaq::CountsDictionary{}}});
      continue;
    }

    simulator.setToZeroState();

    auto mergedTasks = mergeTasksWithTrajectory<ScalarType>(baseTasks, traj,
                                                            batch.noise_sites);

    for (const auto &task : mergedTasks)
      simulator.applyGate(task);
    simulator.flushGateQueue();

    auto execResult =
        simulator.sample(batch.measureQubits, static_cast<int>(traj.num_shots));

    results.push_back(
        cudaq::sample_result{cudaq::ExecutionResult{execResult.counts}});
  }

  return results;
}

// Explicit instantiations for samplePTSBEGeneric
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<float> &, const PTSBatch &);
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<double> &, const PTSBatch &);

/// @brief Helper template for compile-time concept dispatch
template <typename SimulatorType>
std::vector<cudaq::sample_result> dispatchPTSBE(SimulatorType &sim,
                                                const PTSBatch &batch) {

  // Check if it is a BatchSimulator implementation
  auto *batchSim = dynamic_cast<BatchSimulator *>(&sim);
  if (batchSim) {
    return batchSim->sampleWithPTSBE(batch);
  } else {
    return samplePTSBEGeneric(sim, batch);
  }
}

std::vector<cudaq::sample_result> samplePTSBE(const PTSBatch &batch) {
  auto *baseSim = nvqir::getCircuitSimulatorInternal();

  if (baseSim->isSinglePrecision()) {
    auto *sim = dynamic_cast<nvqir::CircuitSimulatorBase<float> *>(baseSim);
    if (!sim)
      throw std::runtime_error(
          "Failed to cast simulator to CircuitSimulatorBase<float>");
    return dispatchPTSBE(*sim, batch);
  } else {
    auto *sim = dynamic_cast<nvqir::CircuitSimulatorBase<double> *>(baseSim);
    if (!sim)
      throw std::runtime_error(
          "Failed to cast simulator to CircuitSimulatorBase<double>");
    return dispatchPTSBE(*sim, batch);
  }
}

std::vector<cudaq::sample_result>
samplePTSBEWithLifecycle(const PTSBatch &batch,
                         const std::string &contextType) {
  auto *sim = nvqir::getCircuitSimulatorInternal();

  cudaq::ExecutionContext ctx(contextType, batch.totalShots());
  cudaq::detail::setExecutionContext(&ctx);
  sim->configureExecutionContext(ctx);
  sim->allocateQubits(batch.kernelTrace.getNumQudits());

  auto results = samplePTSBE(batch);

  // Finalize and reset execution context before deallocating qubits.
  // CircuitSimulatorBase::deallocateQubits is a no-op while an execution
  // context is set, so we must clear it first to avoid leaking qubits.
  sim->finalizeExecutionContext(ctx);
  cudaq::detail::resetExecutionContext();

  std::vector<std::size_t> qubitIds(batch.kernelTrace.getNumQudits());
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  sim->deallocateQubits(qubitIds);

  return results;
}

} // namespace cudaq::ptsbe
