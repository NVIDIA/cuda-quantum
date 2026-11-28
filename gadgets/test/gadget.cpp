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
      get_shape(Opcode::CREATE_PHANTOM) != Shape{0, 1, 0, 0} ||
      get_shape(Opcode::DESTROY_PHANTOM) != Shape{1, 0, 0, 0} ||
      find_opcode("CNOT") != Opcode::CX || to_string(Opcode::CX) != "CX" ||
      find_opcode("MPP"))
    return 1;

  using P = Pauli;
  SubsystemCode rep3(
      "rep3", 3,
      {{false, {{P::Z, 0}, {P::Z, 1}}}, {false, {{P::Z, 1}, {P::Z, 2}}}},
      {0, 1},
      {{{false, {{P::X, 0}, {P::X, 1}, {P::X, 2}}}, {false, {{P::Z, 0}}}}});

  PhysicalInstructionSet native;
  native.name = "native";
  SemanticCircuit x_action({{"a"}, {"b"}});
  x_action.append(OpcodeNode(Opcode::X, {{"a"}, {"b"}}));
  native.append(PhysicalInstructionDefinition("X", std::move(x_action)));
  SemanticCircuit cz_action({{"a", "b"}, {"c", "d"}});
  cz_action.append(OpcodeNode(Opcode::CZ, {{"a", "b"}, {"c", "d"}}));
  native.append(PhysicalInstructionDefinition("CZ", std::move(cz_action)));

  // Realization on rep3's three qubits; logical action the identity on one.
  InstructionCircuit realization(native,
                                 {{"q0", "q2", "q5"}, {"q3", "q4", "q5"}});
  realization.append(InstructionNode(*native.index_of("X"), {{"q0"}, {"q1"}}));
  realization.append(
      InstructionNode(*native.index_of("CZ"), {{"q1", "q2"}, {"q3", "q4"}}));
  SemanticCircuit identity({{"l"}, {"l"}});
  InstructionCircuit empty(native);
  auto rejects_gadget = [&](SemanticCircuit action,
                            InstructionCircuit physical) {
    try {
      Gadget("bad", {&rep3}, {&rep3}, std::move(action), std::move(physical));
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  };
  if (!rejects_gadget(identity, empty) ||
      !rejects_gadget(SemanticCircuit(), realization))
    return 1;
  Gadget gadget("identity", {&rep3}, {&rep3}, identity, std::move(realization),
                {{"o0", {{"m0", "v0"}, true}}});

  GadgetContract contract(gadget, {{{"m0"}, false}});
  if (&contract.get_target() != &gadget ||
      contract.get_detectors().size() != 1 ||
      gadget.get_outcome_map().size() != 1)
    return 1;
  // Each native instruction's action is one semantic operation.
  auto action_opcode = [&](std::size_t i) {
    std::size_t index =
        gadget.get_realization().get_nodes()[i].get_instruction();
    return native.get_instructions()[index]
        .get_action()
        .get_nodes()
        .front()
        .get_opcode();
  };
  return action_opcode(0) == Opcode::X && action_opcode(1) == Opcode::CZ ? 0
                                                                         : 1;
}
