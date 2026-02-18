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
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

template <typename ScalarType>
GateTask<ScalarType> convertToSimulatorTask(const TraceInstruction &inst) {
  std::vector<ScalarType> typedParams;
  typedParams.reserve(inst.params.size());
  for (auto p : inst.params)
    typedParams.push_back(static_cast<ScalarType>(p));

  auto gateName = nvqir::getGateNameFromString(inst.name);
  auto matrix = nvqir::getGateByName<ScalarType>(gateName, typedParams);

  return GateTask<ScalarType>(inst.name, matrix, inst.controls, inst.targets,
                              typedParams);
}

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
convertTrace(std::span<const TraceInstruction> ptsbeTrace) {
  std::vector<GateTask<ScalarType>> tasks;
  tasks.reserve(ptsbeTrace.size());
  for (const auto &inst : ptsbeTrace) {
    if (inst.type == TraceInstructionType::Noise)
      continue;
    if (inst.type == TraceInstructionType::Measurement)
      continue;
    tasks.push_back(convertToSimulatorTask<ScalarType>(inst));
  }
  return tasks;
}

template <typename ScalarType>
GateTask<ScalarType> krausSelectionToTask(const cudaq::KrausSelection &sel,
                                          const TraceInstruction &noiseInst) {
  const auto &channel = noiseInst.channel.value();
  auto k = static_cast<std::size_t>(sel.kraus_operator_index);
  const auto &unitaryDouble = channel.unitary_ops.at(k);
  std::vector<std::complex<ScalarType>> matrix;
  matrix.reserve(unitaryDouble.size());
  for (const auto &elem : unitaryDouble)
    matrix.emplace_back(static_cast<ScalarType>(elem.real()),
                        static_cast<ScalarType>(elem.imag()));
  std::string opName;
  if (k < channel.op_names.size())
    opName = channel.op_names[k];
  else
    opName = channel.get_type_name() + "[" + std::to_string(k) + "]";
  return GateTask<ScalarType>(opName, matrix, {}, sel.qubits, {});
}

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory) {
  const auto &selections = trajectory.kraus_selections;

  std::vector<GateTask<ScalarType>> merged;
  merged.reserve(ptsbeTrace.size());

  std::size_t noiseIdx = 0;
  for (std::size_t i = 0; i < ptsbeTrace.size(); ++i) {
    const auto &inst = ptsbeTrace[i];

    if (inst.type == TraceInstructionType::Gate)
      merged.push_back(convertToSimulatorTask<ScalarType>(inst));

    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == i) {
      merged.push_back(
          krausSelectionToTask<ScalarType>(selections[noiseIdx], inst));
      ++noiseIdx;
    }
  }

  if (noiseIdx < selections.size()) {
    throw std::runtime_error(
        "Invalid circuit_location: " +
        std::to_string(selections[noiseIdx].circuit_location) +
        " >= " + std::to_string(ptsbeTrace.size()));
  }

  return merged;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations for float and double
// ---------------------------------------------------------------------------

template GateTask<float>
convertToSimulatorTask<float>(const TraceInstruction &);
template GateTask<double>
convertToSimulatorTask<double>(const TraceInstruction &);

template std::vector<GateTask<float>>
    convertTrace<float>(std::span<const TraceInstruction>);
template std::vector<GateTask<double>>
    convertTrace<double>(std::span<const TraceInstruction>);

template GateTask<float>
krausSelectionToTask<float>(const cudaq::KrausSelection &,
                            const TraceInstruction &);
template GateTask<double>
krausSelectionToTask<double>(const cudaq::KrausSelection &,
                             const TraceInstruction &);

template std::vector<GateTask<float>>
mergeTasksWithTrajectory<float>(std::span<const TraceInstruction>,
                                const cudaq::KrausTrajectory &);
template std::vector<GateTask<double>>
mergeTasksWithTrajectory<double>(std::span<const TraceInstruction>,
                                 const cudaq::KrausTrajectory &);

// ---------------------------------------------------------------------------
// Non-template implementations
// ---------------------------------------------------------------------------

std::size_t PTSBatch::totalShots() const {
  std::size_t total = 0;
  for (const auto &traj : trajectories)
    total += traj.num_shots;
  return total;
}

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

  std::vector<cudaq::sample_result> results;
  results.reserve(batch.trajectories.size());

  for (const auto &traj : batch.trajectories) {
    if (traj.num_shots == 0) {
      results.push_back(cudaq::sample_result{
          cudaq::ExecutionResult{cudaq::CountsDictionary{}}});
      continue;
    }

    simulator.setToZeroState();

    auto mergedTasks = mergeTasksWithTrajectory<ScalarType>(batch.trace, traj);

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
  sim->allocateQubits(numQubits(batch.trace));

  auto results = samplePTSBE(batch);

  // Finalize and reset execution context before deallocating qubits.
  // CircuitSimulatorBase::deallocateQubits is a no-op while an execution
  // context is set, so we must clear it first to avoid leaking qubits.
  sim->finalizeExecutionContext(ctx);
  cudaq::detail::resetExecutionContext();

  std::vector<std::size_t> qubitIds(numQubits(batch.trace));
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  sim->deallocateQubits(qubitIds);

  return results;
}

} // namespace cudaq::ptsbe
