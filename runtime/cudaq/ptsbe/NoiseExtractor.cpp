/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "NoiseExtractor.h"
#include <sstream>

namespace cudaq::ptsbe::detail {

namespace {

void throwUnitaryMixtureError(const std::string &channel_name,
                              std::size_t trace_index) {
  std::ostringstream msg;
  msg << "Noise channel '" << channel_name << "' at trace index " << trace_index
      << " is not a valid unitary mixture. "
         "PTSBE requires all channels to be unitary mixtures.";
  throw std::invalid_argument(msg.str());
}

} // namespace

NoiseExtractionResult
extractNoiseSites(std::span<const TraceInstruction> ptsbeTrace,
                  bool validate_unitary_mixture) {
  NoiseExtractionResult result;
  result.total_instructions = ptsbeTrace.size();
  result.noisy_instructions = 0;
  result.all_unitary_mixtures = true;

  for (std::size_t i = 0; i < ptsbeTrace.size(); ++i) {
    const auto &inst = ptsbeTrace[i];
    if (inst.type != TraceInstructionType::Noise)
      continue;

    if (!inst.channel.has_value() || inst.channel->empty())
      continue;

    const auto &channel = inst.channel.value();
    if (!channel.is_unitary_mixture()) {
      result.all_unitary_mixtures = false;
      if (validate_unitary_mixture)
        throwUnitaryMixtureError(inst.name, i);
    }

    NoisePoint point;
    point.circuit_location = i;
    point.op_name = inst.name;
    point.qubits = inst.targets;
    point.channel = channel;
    result.noise_sites.push_back(std::move(point));
    result.noisy_instructions++;
  }

  return result;
}

} // namespace cudaq::ptsbe::detail
