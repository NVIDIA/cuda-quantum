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
#include "strategies/ProbabilisticSamplingStrategy.h"
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
        "PTSBE does not support dynamic circuits. "
        "Circuits with conditional logic based on measurement outcomes "
        "cannot currently be pre-trajectory sampled. The gate sequence must be "
        "deterministic for trajectory generation.");
  }
}

bool hasMidCircuitMeasurements(const ExecutionContext &ctx) {
  return !ctx.registerNames.empty();
}

void throwIfMidCircuitMeasurements(const ExecutionContext &ctx) {
  if (hasMidCircuitMeasurements(ctx)) {
    throw std::runtime_error(
        "PTSBE does not support mid-circuit measurements. "
        "Circuits with conditional logic based on measurement outcomes "
        "cannot be pre-trajectory sampled.");
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
        "PTSBE requires a noise model to be set. Please provide a noise "
        "model in sample_options.");
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

PTSBETrace buildPTSBETraceInstructions(const cudaq::Trace &kernelTrace,
                                       const noise_model &noiseModel) {
  PTSBETrace trace;

  auto noiseResult = extractNoiseSites(kernelTrace, noiseModel);

  // Build lookup: gate index -> noise site indices (preserving extraction
  // order)
  std::unordered_map<std::size_t, std::vector<std::size_t>> gateToNoiseSites;
  for (std::size_t i = 0; i < noiseResult.noise_sites.size(); ++i)
    gateToNoiseSites[noiseResult.noise_sites[i].circuit_location].push_back(i);

  // Interleave Gate and Noise instructions
  std::size_t gateIdx = 0;
  for (const auto &inst : kernelTrace) {
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

void populatePTSBETraceTrajectories(
    PTSBETrace &trace, std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults) {
  // TODO: When trajectory generation is wired up, remap circuit_location
  // on each KrausSelection to the corresponding Noise instruction index in
  // trace.instructions and populate measurement_counts from
  // perTrajectoryResults.
  trace.trajectories = std::move(trajectories);
}

PTSBatch buildPTSBatchWithTrajectories(cudaq::Trace &&kernelTrace,
                                       const noise_model &noiseModel,
                                       const PTSBEOptions &options,
                                       std::size_t shots) {
  // TODO: Wire up noise extraction pipeline:
  //   1. extractNoiseSites(kernelTrace, noiseModel)
  //   2. strategy->generateTrajectories(noise_sites, shots)
  //   3. allocateShots(trajectories, shots, ShotAllocationStrategy{})

  PTSBatch batch;
  batch.measureQubits = extractMeasureQubits(kernelTrace);
  batch.kernelTrace = std::move(kernelTrace);
  return batch;
}

} // namespace cudaq::ptsbe
