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
#include <algorithm>
#include <numeric>
#include <span>
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

  // noise_model is optional: noise can come from the model (gate-based) and/or
  // from cudaq.apply_noise() in the kernel.
}

std::vector<std::size_t>
extractMeasureQubits(std::span<const TraceInstruction> trace) {
  std::vector<std::size_t> qubits;
  for (const auto &inst : trace) {
    if (inst.type != TraceInstructionType::Measurement)
      continue;
    for (auto id : inst.targets) {
      if (std::find(qubits.begin(), qubits.end(), id) == qubits.end())
        qubits.push_back(id);
    }
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

static TraceInstructionType
convertInstructionType(cudaq::TraceInstructionType type) {
  switch (type) {
  case cudaq::TraceInstructionType::Gate:
    return TraceInstructionType::Gate;
  case cudaq::TraceInstructionType::Noise:
    return TraceInstructionType::Noise;
  case cudaq::TraceInstructionType::Measurement:
    return TraceInstructionType::Measurement;
  }
  throw std::logic_error("Unknown TraceInstructionType");
}

static std::vector<std::size_t>
extractQubitIds(const std::vector<cudaq::QuditInfo> &qudits) {
  std::vector<std::size_t> ids;
  ids.reserve(qudits.size());
  for (const auto &q : qudits)
    ids.push_back(q.id);
  return ids;
}

static void convertTraceInstruction(const cudaq::Trace::Instruction &inst,
                                    const cudaq::noise_model &noise_model,
                                    std::vector<TraceInstruction> &result) {
  auto targets = extractQubitIds(inst.targets);
  auto controls = extractQubitIds(inst.controls);

  if (inst.type == cudaq::TraceInstructionType::Noise) {
    std::intptr_t key = inst.noise_channel_key.value();
    cudaq::kraus_channel channel = noise_model.get_channel(key, inst.params);
    if (!channel.empty()) {
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      result.push_back({TraceInstructionType::Noise,
                        std::string(cudaq::TRACE_APPLY_NOISE_NAME), targets,
                        controls, inst.params, std::move(channel)});
    }
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Gate) {
    auto channels =
        noise_model.get_channels(inst.name, targets, controls, inst.params);
    result.push_back({TraceInstructionType::Gate, inst.name, targets, controls,
                      inst.params});

    std::vector<std::size_t> noiseQubits = targets;
    noiseQubits.insert(noiseQubits.end(), controls.begin(), controls.end());
    for (auto &channel : channels) {
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      result.push_back({TraceInstructionType::Noise,
                        channel.get_type_name(),
                        noiseQubits,
                        {},
                        {},
                        std::move(channel)});
    }
    return;
  }

  if (inst.type == cudaq::TraceInstructionType::Measurement) {
    auto channels = noise_model.get_channels("mz", targets, {}, {});
    result.push_back({TraceInstructionType::Measurement,
                      inst.name,
                      targets,
                      {},
                      inst.params});

    for (auto &channel : channels) {
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      result.push_back({TraceInstructionType::Noise,
                        channel.get_type_name(),
                        targets,
                        {},
                        {},
                        std::move(channel)});
    }
    return;
  }
}

PTSBETrace buildPTSBETrace(const cudaq::Trace &trace,
                           const cudaq::noise_model &noise_model) {
  PTSBETrace result;
  for (const auto &inst : trace)
    convertTraceInstruction(inst, noise_model, result);
  return result;
}

PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel) {
  PTSBEExecutionData trace;

  auto ptsbeTrace = buildPTSBETrace(kernelTrace, noiseModel);

  // The PTSBE trace already has Gate, Noise, and Measurement interleaved.
  // Validation (unitary mixture checks) happens in the batch execution path.
  for (const auto &inst : ptsbeTrace)
    trace.instructions.push_back(inst);

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

  // 1. Build PTSBE trace, derive measure qubits, extract noise sites
  batch.trace = buildPTSBETrace(kernelTrace, noiseModel);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  auto noiseResult = extractNoiseSites(batch.trace);

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

  return batch;
}

} // namespace cudaq::ptsbe
