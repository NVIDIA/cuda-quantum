/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESample.h"
#include "NoiseExtractor.h"
#include "ShotAllocationStrategy.h"
#include "cudaq/simulators.h"
#include "strategies/ProbabilisticSamplingStrategy.h"
#include <numeric>
#include <unordered_map>

namespace cudaq {
// Forward declaration from cudaq.h
bool kernelHasConditionalFeedback(const std::string &kernelName);
} // namespace cudaq

namespace cudaq::ptsbe {

bool hasConditionalFeedback(const std::string &kernelName,
                            const ExecutionContext &ctx) {
  // Check MLIR-compiled kernel metadata first
  if (cudaq::kernelHasConditionalFeedback(kernelName))
    return true;

  // Fallback: check library mode detection via registerNames
  return !ctx.registerNames.empty();
}

void validatePTSBEKernel(const std::string &kernelName,
                         const ExecutionContext &ctx) {
  if (hasConditionalFeedback(kernelName, ctx)) {
    throw std::runtime_error(
        "PTSBE does not support mid-circuit measurements or dynamic circuits. "
        "Kernel '" +
        kernelName +
        "' contains conditional logic based on measurement outcomes. "
        "The gate sequence must be deterministic for pre-trajectory sampling.");
  }
}

void validatePTSBEPreconditions(quantum_platform &platform,
                                std::optional<std::size_t> qpu_id) {
  if (qpu_id && platform.is_remote(*qpu_id))
    throw std::runtime_error(
        "PTSBE does not support remote execution. Use a local simulator.");

  if (!platform.is_simulator())
    throw std::runtime_error("PTSBE is only supported on simulators.");

  const auto *noise = platform.get_noise();
  if (!noise || noise->empty())
    throw std::runtime_error(
        "PTSBE requires a non-empty noise model. "
        "Pass noise_model=... to cudaq.ptsbe.sample() or set noise on "
        "the platform before calling ptsbe::sample().");
}

std::vector<std::size_t> extractMeasureQubits(const Trace &trace) {
  std::vector<std::size_t> qubits;
  auto numQubits = trace.getNumQudits();
  qubits.reserve(numQubits);
  for (std::size_t i = 0; i < numQubits; ++i) {
    qubits.push_back(i);
  }
  return qubits;
}

void cleanupTracerQubits(const Trace &kernelTrace) {
  auto numQubits = kernelTrace.getNumQudits();
  if (numQubits == 0)
    return;
  std::vector<std::size_t> qubitIds(numQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  cudaq::get_simulator()->deallocateQubits(qubitIds);
}

PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel) {
  PTSBEExecutionData trace;

  auto noiseResult = extractNoiseSites(kernelTrace, noiseModel);

  // Build lookup: gate index -> noise site indices (preserving extraction
  // order)
  std::unordered_map<std::size_t, std::vector<std::size_t>> gateToNoiseSites;
  for (std::size_t i = 0; i < noiseResult.noise_sites.size(); ++i)
    gateToNoiseSites[noiseResult.noise_sites[i].circuit_location].push_back(i);

  // Interleave Gate and Noise instructions
  std::size_t gateIdx = 0;
  for (const auto &inst : kernelTrace) {
    if (inst.noise_channel_key.has_value())
      continue;

    std::vector<std::size_t> targets;
    targets.reserve(inst.targets.size());
    for (const auto &q : inst.targets)
      targets.push_back(q.id);

    std::vector<std::size_t> controls;
    controls.reserve(inst.controls.size());
    for (const auto &q : inst.controls)
      controls.push_back(q.id);

    trace.instructions.push_back(
        TraceInstruction{TraceInstructionType::Gate, inst.name,
                         std::move(targets), std::move(controls), inst.params});

    auto it = gateToNoiseSites.find(gateIdx);
    if (it != gateToNoiseSites.end()) {
      for (auto noiseSiteIdx : it->second) {
        const auto &ns = noiseResult.noise_sites[noiseSiteIdx];
        trace.instructions.push_back(
            TraceInstruction{TraceInstructionType::Noise,
                             ns.channel.get_type_name(),
                             ns.qubits,
                             {},
                             {},
                             ns.channel});
      }
    }

    ++gateIdx;
  }

  auto measureQubits = extractMeasureQubits(kernelTrace);
  for (auto qubit : measureQubits)
    trace.instructions.push_back(TraceInstruction{
        TraceInstructionType::Measurement, "mz", {qubit}, {}, {}});

  return trace;
}

void populateExecutionDataTrajectories(
    PTSBEExecutionData &executionData,
    std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults) {
  // Populate measurement_counts from parallel-indexed perTrajectoryResults
  for (std::size_t i = 0;
       i < trajectories.size() && i < perTrajectoryResults.size(); ++i)
    trajectories[i].measurement_counts = perTrajectoryResults[i].to_map();

  if (!trajectories.empty()) {
    executionData.trajectories = std::move(trajectories);
    return;
  }

  // Stub: generate a single identity trajectory so that the execution data
  // has at least one trajectory for downstream consumers (Python bindings,
  // tests). This will be replaced once the trajectory generation pipeline is
  // wired up.
  KrausTrajectory stub;
  stub.trajectory_id = 0;
  stub.probability = 1.0;
  stub.num_shots = 1;
  for (std::size_t i = 0; i < executionData.instructions.size(); ++i) {
    if (executionData.instructions[i].type == TraceInstructionType::Noise) {
      stub.kraus_selections.emplace_back(
          i, std::vector<std::size_t>(executionData.instructions[i].targets),
          executionData.instructions[i].name, KrausOperatorType::IDENTITY);
    }
  }
  executionData.trajectories.push_back(std::move(stub));
}

PTSBatch buildPTSBatchWithTrajectories(cudaq::Trace &&kernelTrace,
                                       const noise_model &noiseModel,
                                       const PTSBEOptions &options,
                                       std::size_t shots) {
  PTSBatch batch;
  batch.measureQubits = extractMeasureQubits(kernelTrace);

  // 1. Extract noise sites from the trace and noise model
  auto noiseResult = extractNoiseSites(kernelTrace, noiseModel);

  // 2. Generate trajectories via the configured strategy (or default)
  auto strategy = options.strategy
                      ? options.strategy
                      : std::make_shared<ProbabilisticSamplingStrategy>();
  std::size_t maxTrajs = options.max_trajectories.value_or(shots);
  batch.trajectories =
      strategy->generateTrajectories(noiseResult.noise_sites, maxTrajs);

  // 3. Allocate shots across trajectories
  if (!batch.trajectories.empty() && shots > 0)
    allocateShots(batch.trajectories, shots, options.shot_allocation);

  batch.noise_sites = std::move(noiseResult.noise_sites);
  batch.kernelTrace = std::move(kernelTrace);
  return batch;
}

} // namespace cudaq::ptsbe
