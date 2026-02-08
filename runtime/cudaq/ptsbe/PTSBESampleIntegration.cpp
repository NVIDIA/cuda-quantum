/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PTSBESampleIntegration.h"
#include "NoiseExtractor.h"
#include "ShotAllocationStrategy.h"
#include "strategies/ProbabilisticSamplingStrategy.h"

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
