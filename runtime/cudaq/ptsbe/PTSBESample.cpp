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
#include "cudaq/algorithms/sample.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include "strategies/ProbabilisticSamplingStrategy.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <span>
#include <unordered_map>

namespace cudaq::ptsbe::detail {

void validatePTSBEKernel(const std::string &kernelName,
                         const ExecutionContext &ctx) {
  if (cudaq::detail::hasConditionalFeedback(kernelName, &ctx)) {
    throw std::runtime_error(
        "PTSBE does not support mid-circuit measurements or dynamic circuits. "
        "Kernel '" +
        kernelName +
        "' contains conditional logic based on measurement outcomes. "
        "The gate sequence must be deterministic for pre-trajectory sampling.");
  }
}

void warnNamedRegisters(const std::string &kernelName, ExecutionContext &ctx) {
  if (ctx.warnedNamedMeasurements)
    return;
  for (const auto &inst : ctx.kernelTrace) {
    if (inst.type == cudaq::TraceInstructionType::Measurement &&
        inst.register_name && *inst.register_name != "__global__") {
      ctx.warnedNamedMeasurements = true;
      std::cerr << "WARNING: Kernel \"" << kernelName
                << "\" uses named measurement results but is invoked via "
                   "ptsbe::sample (or ptsbe.sample). PTSBE outputs a single "
                   "global register; "
                   "named sub-registers are not preserved. Use `cudaq::run` "
                   "to retrieve individual measurement results."
                << std::endl;
      return;
    }
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
  bool hasMeasurement = false;
  for (const auto &inst : trace) {
    if (inst.type == cudaq::TraceInstructionType::Measurement)
      hasMeasurement = true;
    convertTraceInstruction(inst, noise_model, result);
  }

  // Match standard cudaq::sample() behavior: when the kernel omits explicit
  // mz() calls, measure all allocated qubits. Generate one Measurement + Noise
  // pair per qubit so that per-qubit noise channels (registered via
  // add_channel("mz", {q}, ...)) are matched correctly.
  auto n = trace.getNumQudits();
  if (!hasMeasurement && n > 0) {
    for (std::size_t q = 0; q < n; ++q) {
      result.push_back({TraceInstructionType::Measurement, "mz", {q}, {}, {}});

      auto channels = noise_model.get_channels("mz", {q}, {}, {});
      for (auto &channel : channels) {
        if (channel.empty())
          continue;
        if (!channel.is_unitary_mixture())
          channel.generateUnitaryParameters();
        result.push_back({TraceInstructionType::Noise,
                          channel.get_type_name(),
                          {q},
                          {},
                          {},
                          std::move(channel)});
      }
    }
  }

  return result;
}

PTSBEExecutionData
buildExecutionDataInstructions(const cudaq::Trace &kernelTrace,
                               const noise_model &noiseModel) {
  PTSBEExecutionData trace;

  trace.instructions = buildPTSBETrace(kernelTrace, noiseModel);
  return trace;
}

void populateExecutionDataTrajectories(
    PTSBEExecutionData &executionData,
    std::vector<cudaq::KrausTrajectory> trajectories,
    std::vector<cudaq::sample_result> perTrajectoryResults) {
  // Populate measurement_counts from parallel-indexed perTrajectoryResults,
  // keeping only trajectories that received at least one shot. Zero-shot
  // trajectories were discovered by MC sampling but never simulated.
  for (std::size_t i = 0;
       i < trajectories.size() && i < perTrajectoryResults.size(); ++i) {
    if (trajectories[i].num_shots == 0)
      continue;
    if (perTrajectoryResults[i].get_total_shots() > 0)
      trajectories[i].measurement_counts = perTrajectoryResults[i].to_map();
    executionData.trajectories.push_back(std::move(trajectories[i]));
  }
}

PTSBatch buildPTSBatchFromTrace(PTSBETrace &&trace, const PTSBEOptions &options,
                                std::size_t shots) {
  PTSBatch batch;

  batch.trace = std::move(trace);
  batch.measureQubits = extractMeasureQubits(batch.trace);
  auto noiseResult = extractNoiseSites(batch.trace);
  cudaq::info("[ptsbe] Extracted {} noise sites from {} total instructions",
              noiseResult.noise_sites.size(), noiseResult.total_instructions);

  auto strategy = options.strategy
                      ? options.strategy
                      : std::make_shared<ProbabilisticSamplingStrategy>();
  std::size_t maxTrajs = options.max_trajectories.value_or(shots);
  cudaq::info("[ptsbe] Generating trajectories via {} strategy (max {})",
              strategy->name(), maxTrajs);
  batch.trajectories =
      strategy->generateTrajectories(noiseResult.noise_sites, maxTrajs);

  if (!batch.trajectories.empty() && shots > 0)
    allocateShots(batch.trajectories, shots, options.shot_allocation);

  return batch;
}

} // namespace cudaq::ptsbe::detail
