/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/gadgets/gadget/gadget.h"
#include "cudaq/gadgets/circuit/standard.h"

#include <stdexcept>

using namespace cudaq::gadgets;

int main() {
  auto rejects = [](Opcode opcode, std::optional<Payload> payload) {
    try {
      get_shape(opcode, payload);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  Shape cx = get_shape(Opcode::CX);
  Shape prep = get_shape(Opcode::PREPARE_Z);
  Shape measure = get_shape(Opcode::MEASURE_Z);
  if (cx.q_in != 2 || cx.q_out != 2 || cx.c_out != 0 || prep.q_in != 0 ||
      prep.q_out != 1 || measure.q_out != 0 || measure.c_out != 1 ||
      get_shape(Opcode::RX, ParamValue(0.5)).q_in != 1 ||
      !rejects(Opcode::RX, std::nullopt) ||
      !rejects(Opcode::X, ParamValue(0.5)) ||
      !rejects(Opcode::UNITARY, std::nullopt) ||
      !rejects(Opcode::UNITARY,
               DiagonalUnitary(std::vector<std::complex<double>>(4))) ||
      get_shape(Opcode::UNITARY, Unitary(std::vector<std::complex<double>>(16)))
              .q_in != 2 ||
      get_shape(Opcode::PAULI_MEASUREMENT, PauliString::from_string("ZZZ"))
              .c_out != 3 ||
      find_opcode("CNOT") != Opcode::CX || to_string(Opcode::CX) != "CX" ||
      find_opcode("MPP"))
    return 1;

  SubsystemCode rep3;
  rep3.name = "rep3";

  PhysicalInstructionSet native;
  native.name = "native";
  SemanticCircuit x_action({{"a"}, {"b"}});
  x_action.append(OpcodeNode(Opcode::X, {{"a"}, {"b"}}));
  native.append(PhysicalInstructionDefinition("X", std::move(x_action)));
  SemanticCircuit cz_action({{"a", "b"}, {"c", "d"}});
  cz_action.append(OpcodeNode(Opcode::CZ, {{"a", "b"}, {"c", "d"}}));
  native.append(PhysicalInstructionDefinition("CZ", std::move(cz_action)));

  Gadget gadget;
  gadget.name = "identity";
  gadget.in_codes = {&rep3};
  gadget.out_codes = {&rep3};
  InstructionCircuit realization(native, {{"q0", "q2"}, {"q3", "q4"}});
  realization.append(InstructionNode(*native.index_of("X"), {{"q0"}, {"q1"}}));
  realization.append(
      InstructionNode(*native.index_of("CZ"), {{"q1", "q2"}, {"q3", "q4"}}));
  gadget.realization = std::move(realization);
  gadget.boundary_bindings.push_back({0, false, {"q0", "q1", "q2"}, {"l0"}});
  gadget.outcome_map.push_back({"o0", {{"m0", "v0"}, true}});

  GadgetSet gadgets;
  gadgets.append(std::move(gadget));
  const Gadget *stored = gadgets.find("identity");
  if (!stored)
    return 1;
  // Each native instruction's action is one semantic operation.
  auto action_opcode = [&](std::size_t i) {
    std::size_t index = std::get<InstructionCircuit>(stored->realization)
                            .get_nodes()[i]
                            .get_instruction();
    return native.get_instructions()[index]
        .get_action()
        .get_nodes()
        .front()
        .get_opcode();
  };
  return action_opcode(0) == Opcode::X && action_opcode(1) == Opcode::CZ ? 0
                                                                         : 1;
}
