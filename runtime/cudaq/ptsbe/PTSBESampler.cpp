/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESample.h"
#include "PTSBESamplerImpl.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "cudaq/ptsbe/policy.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>

namespace cudaq::ptsbe {

template <typename ScalarType>
std::vector<GateTask<ScalarType>>
mergeTasksWithTrajectory(std::span<const TraceInstruction> ptsbeTrace,
                         const cudaq::KrausTrajectory &trajectory,
                         bool includeIdentity) {
  const auto &selections = trajectory.kraus_selections;

  std::vector<GateTask<ScalarType>> merged;
  merged.reserve(ptsbeTrace.size());

  std::size_t noiseIdx = 0;
  for (std::size_t i = 0; i < ptsbeTrace.size(); ++i) {
    const auto &inst = ptsbeTrace[i];

    if (inst.type == TraceInstructionType::Gate)
      merged.push_back(detail::convertToSimulatorTask<ScalarType>(inst));

    while (noiseIdx < selections.size() &&
           selections[noiseIdx].circuit_location == i) {
      if (includeIdentity || selections[noiseIdx].is_error) {
        merged.push_back(detail::krausSelectionToTask<ScalarType>(
            selections[noiseIdx], inst));
      }
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

template std::vector<GateTask<float>>
mergeTasksWithTrajectory<float>(std::span<const TraceInstruction>,
                                const cudaq::KrausTrajectory &, bool);
template std::vector<GateTask<double>>
mergeTasksWithTrajectory<double>(std::span<const TraceInstruction>,
                                 const cudaq::KrausTrajectory &, bool);

std::size_t PTSBatch::totalShots() const {
  std::size_t total = 0;
  for (const auto &traj : trajectories)
    total += traj.num_shots;
  return total;
}

} // namespace cudaq::ptsbe

namespace cudaq::ptsbe::detail {

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
  if (noiseInst.type != TraceInstructionType::Noise || !noiseInst.channel)
    throw std::runtime_error(
        "krausSelectionToTask: expected Noise instruction with a channel at "
        "circuit_location " +
        std::to_string(sel.circuit_location));
  const auto &channel = noiseInst.channel.value();
  auto k = sel.kraus_operator_index;
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

cudaq::sample_result
aggregateResults(const std::vector<cudaq::sample_result> &results) {
  if (results.empty())
    return cudaq::sample_result{};

  cudaq::CountsDictionary aggregatedCounts;
  std::vector<std::string> aggregatedSeqData;
  for (const auto &res : results) {
    if (res.get_total_shots() == 0)
      continue;

    for (const auto &[bitstring, count] : res.to_map())
      aggregatedCounts[bitstring] += count;

    auto seq = res.sequential_data();
    if (!seq.empty())
      aggregatedSeqData.insert(aggregatedSeqData.end(),
                               std::make_move_iterator(seq.begin()),
                               std::make_move_iterator(seq.end()));
  }
  cudaq::ExecutionResult er{aggregatedCounts};
  er.sequentialData = std::move(aggregatedSeqData);
  return cudaq::sample_result{std::move(er)};
}

template <typename ScalarType>
std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<ScalarType> &simulator,
                   const PTSBatch &batch) {
  ScopedTraceWithContext("ptsbe::samplePTSBEGeneric",
                         batch.trajectories.size());
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

  const std::size_t numTrajectories = batch.trajectories.size();
  // Log progress at ~10% intervals (at least every 100 trajectories)
  const std::size_t progressInterval = std::max<std::size_t>(
      1, std::min<std::size_t>(numTrajectories / 10, 100));

  for (std::size_t ti = 0; ti < numTrajectories; ++ti) {
    const auto &traj = batch.trajectories[ti];
    if (traj.num_shots == 0) {
      results.push_back(cudaq::sample_result{
          cudaq::ExecutionResult{cudaq::CountsDictionary{}}});
      continue;
    }

    simulator.setToZeroState();

    auto mergedTasks =
        cudaq::ptsbe::mergeTasksWithTrajectory<ScalarType>(batch.trace, traj);

    for (const auto &task : mergedTasks)
      simulator.applyGate(task);
    simulator.flushGateQueue();

    auto execResult =
        simulator.sample(batch.measureQubits, static_cast<int>(traj.num_shots),
                         batch.includeSequentialData);

    cudaq::ExecutionResult er{execResult.counts};
    if (batch.includeSequentialData)
      er.sequentialData = std::move(execResult.sequentialData);
    results.push_back(cudaq::sample_result{std::move(er)});

    if ((ti + 1) % progressInterval == 0)
      cudaq::info("[ptsbe] Trajectory progress: {}/{} ({} shots)", ti + 1,
                  numTrajectories, traj.num_shots);
  }

  return results;
}

// Explicit instantiations for samplePTSBEGeneric
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<float> &, const PTSBatch &);
template std::vector<cudaq::sample_result>
samplePTSBEGeneric(nvqir::CircuitSimulatorBase<double> &, const PTSBatch &);

namespace {

/// @brief Helper template for simulator dispatch
template <typename SimulatorType>
std::vector<cudaq::sample_result> dispatchPTSBE(SimulatorType &sim,
                                                const PTSBatch &batch) {

  // Check env var to force the generic (per-trajectory sampler) path,
  // bypassing the batched batchMeasure path which has a per-shot GPU loop.
  const bool forceGeneric =
      cudaq::getEnvBool("CUDAQ_PTSBE_FORCE_GENERIC", false);

  if (!forceGeneric) {
    auto *batchSim = dynamic_cast<BatchSimulator *>(&sim);
    if (batchSim) {
      cudaq::info(
          "[ptsbe] Dispatching to BatchSimulator custom implementation");
      return batchSim->sampleWithPTSBE(batch);
    }
  } else {
    cudaq::info("[ptsbe] BatchSimulator dispatch overridden by "
                "CUDAQ_PTSBE_FORCE_GENERIC=1");
  }

  cudaq::info("[ptsbe] Dispatching to generic per-trajectory sampler");
  return samplePTSBEGeneric(sim, batch);
}

} // namespace

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

namespace {
template <typename Fn>
void withClearedExecutionContext(Fn &&f) {
  auto *ctx = cudaq::getExecutionContext();
  cudaq::detail::resetExecutionContext();
  try {
    f();
  } catch (...) {
    if (ctx)
      cudaq::detail::setExecutionContext(ctx);
    throw;
  }
  if (ctx)
    cudaq::detail::setExecutionContext(ctx);
}
} // namespace

ptsbe::sample_result finalizePTSBE(const cudaq::ptsbe::sample_policy &policy) {
  if (policy.batch) {
    auto results = samplePTSBE(*policy.batch);
    auto aggregated = aggregateResults(results);
    policy.perTrajectoryResults = std::move(results);
    return ptsbe::sample_result(std::move(aggregated));
  }

  auto *ctx = cudaq::getExecutionContext();
  if (!ctx || ctx->name != "tracer")
    throw std::runtime_error(
        "ptsbe::sample_policy has no PTSBatch attached and no tracer "
        "execution context is active. PTSBE cannot be finalized by name-only "
        "dispatch.");

  // stage 0: normalize tracer qubit bookkeeping under a cleared context
  withClearedExecutionContext([&] { cleanupTracerQubits(ctx->kernelTrace); });
  cudaq::info("[ptsbe] Trace captured: {} qubits, {} instructions",
              ctx->kernelTrace.getNumQudits(),
              ctx->kernelTrace.getNumInstructions());

  // stage 1: validate eligibility (no dynamic circuits) and warn on named
  // registers
  validatePTSBEKernel(policy.kernelName, *ctx);
  warnNamedRegisters(policy.kernelName, *ctx);

  // stage 2: build the PTSBE trace from the captured kernel trace
  static const cudaq::noise_model kEmptyNoiseModel;
  const auto &noiseModel =
      policy.noiseModel ? *policy.noiseModel : kEmptyNoiseModel;
  auto ptsbeTrace = buildPTSBETrace(ctx->kernelTrace, noiseModel);

  std::optional<PTSBEExecutionData> executionData;
  if (policy.options.return_execution_data) {
    executionData = PTSBEExecutionData{};
    executionData->instructions = ptsbeTrace;
  }

  // stage 3: trajectory generation + shot allocation
  auto batch = buildPTSBatchFromTrace(std::move(ptsbeTrace), policy.options,
                                      policy.shots);
  batch.includeSequentialData = policy.options.include_sequential_data;
  cudaq::info("[ptsbe] Allocated {} shots across {} trajectories",
              batch.totalShots(), batch.trajectories.size());

  // stages 4 and 5: replay the trajectories under a dedicated sampling
  // context, distinct from the tracer context that captured the trace.
  cudaq::ExecutionContext sampleCtx(cudaq::ptsbe::sample_policy::name,
                                    batch.totalShots());
  sampleCtx.kernelName = policy.kernelName;

  const auto nQubits = numQubits(batch.trace);
  ptsbe::sample_result result;
  cudaq::detail::with_policy_and_ctx(policy, sampleCtx, [&] {
    nvqir::getCircuitSimulatorInternal()->configureExecutionContext(sampleCtx);
    try {
      allocateBatchQubits(nQubits);
      auto results = samplePTSBE(batch);
      result = ptsbe::sample_result(aggregateResults(results));
      policy.perTrajectoryResults = std::move(results);
    } catch (...) {
      releaseBatchQubits(nQubits);
      throw;
    }
  });

  // stage 6: attach trajectories + execution data if requested
  if (executionData) {
    populateExecutionDataTrajectories(*executionData,
                                      std::move(batch.trajectories),
                                      std::move(policy.perTrajectoryResults));
    result.set_execution_data(std::move(*executionData));
  }

  cudaq::info("[ptsbe] Complete: {} unique bitstrings from {} shots",
              result.size(), result.get_total_shots());
  return result;
}

void allocateBatchQubits(std::size_t nQubits) {
  nvqir::getCircuitSimulatorInternal()->allocateQubits(nQubits);
}

void releaseBatchQubits(std::size_t nQubits) {
  std::vector<std::size_t> qubitIds(nQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  auto *ctx = cudaq::getExecutionContext();
  if (ctx)
    cudaq::detail::resetExecutionContext();
  try {
    nvqir::getCircuitSimulatorInternal()->deallocateQubits(qubitIds);
  } catch (...) {
    if (ctx)
      cudaq::detail::setExecutionContext(ctx);
    throw;
  }
  if (ctx)
    cudaq::detail::setExecutionContext(ctx);
}

} // namespace cudaq::ptsbe::detail
