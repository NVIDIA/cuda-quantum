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
Shape Circuit::get_shape(const Node &node) const {
  if (!instruction_set) {
    if (node.get_opcode() >= num_opcodes)
      throw std::invalid_argument("node '" + node.get_id() +
                                  "': unknown opcode");
    return cudaq::gadgets::get_shape(static_cast<Opcode>(node.get_opcode()),
                                     node.get_payload());
  }
  const auto &instructions = instruction_set->get_instructions();
  if (node.get_opcode() >= instructions.size())
    throw std::invalid_argument("node '" + node.get_id() +
                                "': not in the instruction set");
  if (node.get_payload())
    throw std::invalid_argument("node '" + node.get_id() +
                                "': physical instructions take no payload");
  return instructions[node.get_opcode()].get_shape();
}

std::pair<Id, std::vector<OutcomeRef>> Circuit::append(Node node) {
  if (node.get_shape() != get_shape(node))
    throw std::invalid_argument("node '" + node.get_id() +
                                "': wire counts do not match the operation");
  auto result = std::make_pair(node.get_id(), node.get_c_out());
  nodes.push_back(std::move(node));
  return result;
}

PhysicalInstructionDefinition::PhysicalInstructionDefinition(std::string name,
                                                             Circuit action)
    : name(std::move(name)),
      shape{action.get_q_in().size(), action.get_q_out().size(),
            action.get_c_in().size(), action.get_c_out().size()},
      action(std::move(action)) {
  const Circuit &circuit = this->action;
  if (circuit.get_instruction_set())
    throw std::invalid_argument("instruction '" + this->name +
                                "': action must use semantic instructions");
  circuit.validate();

  // Current wire of each input port; empty once discarded.
  std::vector<WireId> wires = circuit.get_q_in();
  for (const Node &node : circuit.get_nodes())
    for (std::size_t k = 0; k < node.get_q_in().size(); ++k) {
      auto it = std::find(wires.begin(), wires.end(), node.get_q_in()[k]);
      if (it == wires.end())
        continue; // Prepared inside the action.
      // Semantic input k continues as output k, if there is one.
      *it = k < node.get_q_out().size() ? node.get_q_out()[k] : WireId();
    }

  for (std::size_t i = 0; i < wires.size(); ++i)
    if (!wires[i].empty()) // A valid action ends each live input at an output.
      port_map.emplace_back(i, std::find(circuit.get_q_out().begin(),
                                         circuit.get_q_out().end(), wires[i]) -
                                   circuit.get_q_out().begin());
}

} // namespace v0
} // namespace cudaq::gadgets
