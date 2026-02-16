/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "NoiseExtractor.h"
#include <sstream>
#include <vector>

namespace cudaq::ptsbe {

static std::vector<std::size_t>
qubitIdsFromInstruction(const cudaq::Trace::Instruction &inst) {
  std::vector<std::size_t> qubits;
  qubits.reserve(inst.targets.size() + inst.controls.size());
  for (const auto &q : inst.targets) {
    qubits.push_back(q.id);
  }
  for (const auto &q : inst.controls) {
    qubits.push_back(q.id);
  }
  return qubits;
}

static NoisePoint createNoisePoint(std::size_t index,
                                   const std::string &op_name,
                                   std::vector<std::size_t> qubits,
                                   cudaq::kraus_channel channel) {
  NoisePoint point;
  point.circuit_location = index;
  point.op_name = op_name;
  point.qubits = std::move(qubits);
  point.channel = std::move(channel);
  return point;
}

static void throwUnitaryMixtureError(const std::string &gate_name,
                                     std::size_t instruction_idx) {
  std::ostringstream msg;
  msg << "Noise channel for gate '" << gate_name << "' at instruction "
      << instruction_idx
      << " is not a valid unitary mixture. "
         "PTSBE requires all channels to be unitary mixtures.";
  throw std::invalid_argument(msg.str());
}

NoiseExtractionResult extractNoiseSites(const cudaq::Trace &trace,
                                        const cudaq::noise_model &noise_model,
                                        bool validate_unitary_mixture) {
  NoiseExtractionResult result;
  result.total_instructions = trace.getNumInstructions();
  result.noisy_instructions = 0;
  result.all_unitary_mixtures = true;

  std::size_t instruction_idx = 0;

  for (const auto &inst : trace) {
    if (inst.noise_channel_key.has_value()) {
      // Inline apply_noise: insert after the last gate (instruction_idx - 1)
      std::size_t loc = (instruction_idx > 0) ? instruction_idx - 1 : 0;
      std::intptr_t key = *inst.noise_channel_key;
      cudaq::kraus_channel channel = noise_model.get_channel(key, inst.params);
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      if (!channel.is_unitary_mixture())
        throwUnitaryMixtureError(inst.name, loc);
      std::vector<std::size_t> qubits = qubitIdsFromInstruction(inst);
      result.noise_sites.push_back(createNoisePoint(
          loc, inst.name, std::move(qubits), std::move(channel)));
      result.noisy_instructions++;
      continue;
    }

    // Gate: look up channels by gate name and qubits
    std::vector<std::size_t> target_qubits;
    target_qubits.reserve(inst.targets.size());
    for (const auto &q : inst.targets)
      target_qubits.push_back(q.id);
    std::vector<std::size_t> control_qubits;
    control_qubits.reserve(inst.controls.size());
    for (const auto &q : inst.controls)
      control_qubits.push_back(q.id);

    auto channels = noise_model.get_channels(inst.name, target_qubits,
                                             control_qubits, inst.params);
    bool instruction_has_noise = false;

    for (auto channel : channels) {
      if (channel.empty())
        continue;
      if (!channel.is_unitary_mixture())
        channel.generateUnitaryParameters();
      if (!channel.is_unitary_mixture())
        throwUnitaryMixtureError(inst.name, instruction_idx);
      std::vector<std::size_t> qubits = qubitIdsFromInstruction(inst);
      result.noise_sites.push_back(createNoisePoint(
          instruction_idx, inst.name, std::move(qubits), std::move(channel)));
      instruction_has_noise = true;
    }

    if (instruction_has_noise)
      result.noisy_instructions++;

    instruction_idx++;
  }

  return result;
}

} // namespace cudaq::ptsbe
