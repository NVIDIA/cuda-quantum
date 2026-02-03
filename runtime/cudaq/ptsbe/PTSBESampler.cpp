/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESampler.h"
#include "cudaq/simulators.h"
#include <numeric>

namespace cudaq::ptsbe {

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
  if (!simulator.getExecutionContext())
    throw std::runtime_error(
        "samplePTSBEGeneric requires ExecutionContext to be set. "
        "Call simulator.setExecutionContext() before invoking.");

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

    auto mergedTasks = mergeTasksWithTrajectory<ScalarType>(baseTasks, traj);

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

// Explicit instantiations for float and double
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<float> &, const PTSBatch &);
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<double> &, const PTSBatch &);

/// @brief Helper template for compile-time concept dispatch
template <typename SimulatorType>
std::vector<cudaq::sample_result>
dispatchPTSBE(SimulatorType &sim, const PTSBatch &batch) {
  if constexpr (PTSBECapable<SimulatorType>) {
    return sim.sampleWithPTSBE(batch);
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
samplePTSBEWithLifecycle(const PTSBatch &batch, const std::string &contextType) {
  auto *sim = nvqir::getCircuitSimulatorInternal();

  cudaq::ExecutionContext ctx(contextType, batch.totalShots());
  sim->setExecutionContext(&ctx);
  sim->allocateQubits(batch.kernelTrace.getNumQudits());

  auto results = samplePTSBE(batch);

  std::vector<std::size_t> qubitIds(batch.kernelTrace.getNumQudits());
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  sim->deallocateQubits(qubitIds);
  sim->resetExecutionContext();

  return results;
}

} // namespace cudaq::ptsbe
