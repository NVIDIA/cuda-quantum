/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/circuit/circuit.h"

#include <algorithm>

namespace cudaq::gadgets {
inline namespace v0 {
std::size_t InstructionCircuit::append(InstructionNode node) {
  const auto &instructions = instruction_set->get_instructions();
  if (node.get_instruction() >= instructions.size())
    throw std::invalid_argument("instruction " +
                                std::to_string(node.get_instruction()) +
                                " is not in the instruction set");
  node.check_shape(instructions[node.get_instruction()].get_shape());
  nodes.push_back(std::move(node));
  return nodes.size() - 1;
}

PhysicalInstructionDefinition::PhysicalInstructionDefinition(
    std::string name, SemanticCircuit action)
    : name(std::move(name)), shape(action.get_wires().get_shape()),
      action(std::move(action)) {
  const SemanticCircuit &circuit = this->action;
  circuit.validate();

  // Current wire of each input port; empty once discarded.
  const std::vector<WireId> &outputs = circuit.get_wires().q_out;
  std::vector<WireId> wires = circuit.get_wires().q_in;
  for (const OpcodeNode &node : circuit.get_nodes())
    for (std::size_t k = 0; k < node.get_wires().q_in.size(); ++k) {
      auto it = std::find(wires.begin(), wires.end(), node.get_wires().q_in[k]);
      if (it == wires.end())
        continue; // Prepared inside the action.
      // Semantic input k continues as output k, if there is one.
      const std::vector<WireId> &out = node.get_wires().q_out;
      *it = k < out.size() ? out[k] : WireId();
    }

  for (std::size_t i = 0; i < wires.size(); ++i)
    if (!wires[i].empty()) // A valid action ends each live input at an output.
      port_map.emplace_back(
          i, std::find(outputs.begin(), outputs.end(), wires[i]) -
                 outputs.begin());
}

} // namespace v0
} // namespace cudaq::gadgets
