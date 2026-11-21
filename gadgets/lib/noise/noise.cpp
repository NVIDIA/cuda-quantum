/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/noise/noise.h"

namespace cudaq::gadgets {
inline namespace v0 {

InstructionSetNoiseModel::InstructionSetNoiseModel(
    const PhysicalInstructionSet &instruction_set, InstructionNoise noise)
    : instruction_set(&instruction_set), instruction_noise(std::move(noise)) {
  for (const auto &[name, outputs] : instruction_noise) {
    const PhysicalInstructionDefinition *def = instruction_set.find(name);
    if (!def)
      throw std::invalid_argument("no instruction '" + name + "'");
    if (outputs.size() != def->get_q_out().size() + def->get_c_out())
      throw std::invalid_argument("noise for '" + name +
                                  "' needs one entry per output");
    for (std::size_t k = def->get_q_out().size(); k < outputs.size(); ++k)
      if (outputs[k].y || outputs[k].z)
        throw std::invalid_argument("classical outputs of '" + name +
                                    "' only have X noise");
  }
}

CircuitNoise InstructionSetNoiseModel::noise(const Circuit &circuit) const {
  if (circuit.instruction_set != instruction_set)
    throw std::invalid_argument("circuit does not use the instruction set");

  std::unordered_map<WireId, PauliNoise> noise;
  for (const auto &node : circuit.nodes) {
    if (!InstructionNode::classof(node.get()))
      continue;
    auto it = instruction_noise.find(
        static_cast<const InstructionNode &>(*node).get_callee());
    if (it == instruction_noise.end())
      continue;
    std::size_t k = 0;
    for (const WireId &wire : node->get_q_out())
      noise.emplace(wire, it->second[k++]);
    for (const WireId &wire : node->get_c_out())
      noise.emplace(wire, it->second[k++]);
  }
  std::vector<PauliNoise> output_noise;
  for (const auto *wires : {&circuit.q_out, &circuit.c_out})
    for (const WireId &wire : *wires) {
      auto it = noise.find(wire);
      output_noise.push_back(it == noise.end() ? PauliNoise{} : it->second);
    }
  return CircuitNoise(std::move(noise), std::move(output_noise),
                      circuit.q_out.size());
}

} // namespace v0
} // namespace cudaq::gadgets
