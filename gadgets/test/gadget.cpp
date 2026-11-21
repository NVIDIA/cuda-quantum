/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/gadgets/gadget/gadget.h"
#include "cudaq/gadgets/circuit/standard.h"

using namespace cudaq::gadgets;

int main() {
  const OpcodeContract &mzz = contract(Opcode::MZZ);
  const OpcodeContract &prep = contract(Opcode::PREP_Z);
  const OpcodeContract &discard = contract(Opcode::DISCARD);
  if (mzz.q_in != 2 || mzz.q_out != 2 || mzz.c_out != 1 || prep.q_in != 0 ||
      !prep.out_flags.prepared || discard.q_out != 0 ||
      !discard.in_flags.discarded || discard.c_out != 0 ||
      !contract(Opcode::DIAGONAL_ROTATION).param || find_opcode("MPP"))
    return 1;

  SubsystemCode rep3;
  rep3.name = "rep3";

  PhysicalInstructionSet native;
  native.name = "native";
  Circuit x_action;
  x_action.q_in = {"a"};
  x_action.q_out = {"b"};
  x_action.append(std::make_unique<OpcodeNode>(
      "x", Opcode::X, std::vector<WireId>{"a"}, std::vector<WireId>{"b"}));
  native.append(PhysicalInstructionDefinition("X", std::move(x_action)));
  Circuit mzz_action;
  mzz_action.q_in = {"a", "b"};
  mzz_action.q_out = {"c", "d"};
  mzz_action.c_out = {"m"};
  mzz_action.append(std::make_unique<OpcodeNode>(
      "mzz", Opcode::MZZ, std::vector<WireId>{"a", "b"},
      std::vector<WireId>{"c", "d"}, std::vector<WireId>{"m"}));
  native.append(PhysicalInstructionDefinition("MZZ", std::move(mzz_action)));

  Gadget gadget;
  gadget.name = "identity";
  gadget.in_codes = {&rep3};
  gadget.out_codes = {&rep3};
  gadget.realization.instruction_set = &native;
  gadget.realization.nodes.push_back(std::make_unique<InstructionNode>(
      "flip", "X", std::vector<WireId>{"q0"}, std::vector<WireId>{"q1"}));
  gadget.realization.nodes.push_back(std::make_unique<InstructionNode>(
      "measure", "MZZ", std::vector<WireId>{"q1", "q2"},
      std::vector<WireId>{"q3", "q4"}, std::vector<WireId>{},
      std::vector<WireId>{"m0"}));
  gadget.boundary_bindings.push_back({0, false, {"q0", "q1", "q2"}, {"l0"}});
  gadget.outcome_map.push_back({"o0", {{"m0", "v0"}, true}});

  GadgetSet gadgets;
  gadgets.append(std::move(gadget));
  GadgetCircuit program;
  program.instruction_set = &gadgets;
  program.q_in = {"b0"};
  program.nodes.push_back(std::make_unique<GadgetCallNode>(
      "call", "identity", std::vector<WireId>{}, std::vector<WireId>{}));

  const Node *first = program.nodes[0].get();
  if (!GadgetCallNode::classof(first) || InstructionNode::classof(first))
    return 1;
  const Gadget *stored =
      gadgets.find(static_cast<const GadgetCallNode *>(first)->get_callee());
  if (!stored)
    return 1;
  // Each native instruction's action is one standard operation.
  auto action_opcode = [&](std::size_t i) {
    const Node *node = stored->realization.nodes[i].get();
    if (!InstructionNode::classof(node))
      return Opcode::I;
    const Node *op =
        native.find(static_cast<const InstructionNode *>(node)->get_callee())
            ->get_action()
            .nodes.front()
            .get();
    return OpcodeNode::classof(op)
               ? static_cast<const OpcodeNode *>(op)->get_opcode()
               : Opcode::I;
  };
  return action_opcode(0) == Opcode::X && action_opcode(1) == Opcode::MZZ ? 0
                                                                          : 1;
}
