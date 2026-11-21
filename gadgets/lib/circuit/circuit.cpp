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
namespace {

/// The definition node calls.
const PhysicalInstructionDefinition &definition(const Circuit &circuit,
                                                const InstructionNode &node) {
  const PhysicalInstructionDefinition *def =
      circuit.instruction_set ? circuit.instruction_set->find(node.get_callee())
                              : nullptr;
  if (!def)
    throw std::invalid_argument("node '" + node.get_id() + "': instruction '" +
                                node.get_callee() +
                                "' is not in the circuit's instruction set");
  return *def;
}

/// Whether node resets qubit output j.
bool resets(const Circuit &circuit, const Node &node, std::size_t j) {
  if (OpcodeNode::classof(&node))
    return contract(static_cast<const OpcodeNode &>(node).get_opcode())
        .out_flags.reset;
  if (InstructionNode::classof(&node))
    return definition(circuit, static_cast<const InstructionNode &>(node))
        .get_q_out()[j]
        .reset;
  return false;
}

} // namespace

std::optional<std::size_t>
InstructionNode::trace(std::size_t k, const Circuit &circuit) const {
  for (auto [in, out] : definition(circuit, *this).get_port_map())
    if (in == k)
      return out;
  return std::nullopt;
}

PhysicalInstructionDefinition::PhysicalInstructionDefinition(std::string name,
                                                             Circuit action)
    : name(std::move(name)), q_in(action.q_in.size()),
      q_out(action.q_out.size()), c_in(action.c_in.size()),
      c_out(action.c_out.size()), action(std::move(action)) {
  const Circuit &circuit = this->action;
  circuit.validate();
  // TODO: Derive params from the parameter names the action's nodes use.

  // Current wire of each input port.
  std::vector<WireId> wires = circuit.q_in;
  std::vector<bool> reset(wires.size());
  for (const auto &node : circuit.nodes)
    for (std::size_t k = 0; k < node->get_q_in().size(); ++k) {
      auto it = std::find(wires.begin(), wires.end(), node->get_q_in()[k]);
      if (it == wires.end())
        continue; // Prepared inside the action.
      std::size_t i = it - wires.begin();
      if (auto j = node->trace(k, circuit)) {
        *it = node->get_q_out()[*j];
        reset[i] = reset[i] || resets(circuit, *node, *j);
      } else {
        q_in[i].discarded = true;
      }
    }

  std::vector<bool> reached(circuit.q_out.size());
  for (u32 i = 0; i < wires.size(); ++i) {
    if (q_in[i].discarded)
      continue;
    // A valid action ends each live input at an output.
    u32 j = std::find(circuit.q_out.begin(), circuit.q_out.end(), wires[i]) -
            circuit.q_out.begin();
    port_map.emplace_back(i, j);
    reached[j] = true;
    q_out[j].reset = reset[i];
    q_in[i].inactive = q_out[j].inactive = wires[i] == circuit.q_in[i];
  }
  for (u32 j = 0; j < circuit.q_out.size(); ++j)
    q_out[j].prepared = !reached[j];
}

const PhysicalInstructionSet &standard_instruction_set() {
  static const PhysicalInstructionSet set = [] {
    auto wires = [](const char *prefix, u32 count) {
      std::vector<WireId> names;
      for (u32 k = 0; k < count; ++k)
        names.push_back(prefix + std::to_string(k));
      return names;
    };
    PhysicalInstructionSet standard;
    standard.name = "standard";
    for (u32 i = 0; i < num_opcodes; ++i) {
      const OpcodeContract &op = contract(static_cast<Opcode>(i));
      Circuit action;
      action.q_in = wires("in", op.q_in);
      action.q_out = wires("out", op.q_out);
      action.c_out = wires("m", op.c_out);
      std::optional<ParamValue> param;
      if (op.param)
        param = Param(op.param);
      action.append(std::make_unique<OpcodeNode>(
          op.name, op.opcode, action.q_in, action.q_out, action.c_out, param));
      standard.append(
          PhysicalInstructionDefinition(op.name, std::move(action)));
    }
    return standard;
  }();
  return set;
}

} // namespace v0
} // namespace cudaq::gadgets
