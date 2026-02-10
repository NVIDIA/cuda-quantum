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
