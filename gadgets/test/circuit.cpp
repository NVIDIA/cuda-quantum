/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/gadgets/circuit/circuit.h"

#include <cstdio>

using namespace cudaq::gadgets;
using Wires = std::vector<WireId>;

namespace {
int failures = 0;

template <typename F>
void expect_throw(const char *name, F f) {
  try {
    f();
  } catch (const std::invalid_argument &) {
    return;
  }
  std::printf("FAIL: %s did not throw\n", name);
  ++failures;
}

template <typename F>
void expect_ok(const char *name, F f) {
  try {
    f();
  } catch (const std::invalid_argument &e) {
    std::printf("FAIL: %s threw: %s\n", name, e.what());
    ++failures;
  }
}

std::unique_ptr<Node> op(Opcode opcode, Wires q_in, Wires q_out,
                         Wires c_out = {}) {
  return std::make_unique<OpcodeNode>("", opcode, std::move(q_in),
                                      std::move(q_out), std::move(c_out));
}

/// H a; CX a b; M a -> m; X b if m.
Circuit bell_correction() {
  Circuit circuit;
  circuit.q_in = {"a0", "b0"};
  circuit.append(op(Opcode::H, {"a0"}, {"a1"}));
  circuit.append(op(Opcode::CX, {"a1", "b0"}, {"a2", "b1"}));
  auto [id, outcomes] = circuit.append(std::make_unique<OpcodeNode>(
      "measure", Opcode::M, Wires{"a2"}, Wires{"a3"}, Wires{"m"}));
  if (id != "measure" || outcomes != Wires{"m"}) {
    std::printf("FAIL: append result\n");
    ++failures;
  }
  circuit.append(std::make_unique<ConditionalPauliNode>(
      "fix", PauliString::from_string("+X"), Predicate{{"m"}, false},
      Wires{"b1"}, Wires{"b2"}));
  circuit.q_out = {"a3", "b2"};
  circuit.c_out = {"m"};
  return circuit;
}
} // namespace

int main() {
  expect_ok("valid circuit", [] { bell_correction().validate(); });
  expect_ok("random bit and discard", [] {
    Circuit circuit;
    circuit.q_in = {"a"};
    circuit.append(std::make_unique<RandomBitNode>("random", "r"));
    circuit.append(op(Opcode::DISCARD, {"a"}, {}));
    circuit.c_out = {"r"};
    circuit.validate();
  });

  expect_throw("conditional Pauli width", [] {
    Circuit circuit;
    circuit.q_in = {"a"};
    circuit.append(std::make_unique<ConditionalPauliNode>(
        "fix", PauliString::from_string("+ZZ"), Predicate{}, Wires{"a"},
        Wires{"b"}));
  });

  PhysicalInstructionSet native;
  Circuit cz;
  cz.q_in = {"a", "b"};
  cz.q_out = {"c", "d"};
  cz.append(op(Opcode::CZ, {"a", "b"}, {"c", "d"}));
  native.append(PhysicalInstructionDefinition("CZ", std::move(cz)));
  auto call = [&](Wires q_in, Wires q_out) {
    return std::make_unique<InstructionNode>("cz", "CZ", std::move(q_in),
                                             std::move(q_out));
  };
  expect_ok("instruction", [&] {
    Circuit circuit;
    circuit.instruction_set = &native;
    circuit.q_in = {"a", "b"};
    circuit.append(call({"a", "b"}, {"c", "d"}));
    circuit.q_out = {"c", "d"};
    circuit.validate();
  });

  // Signature derived by tracing: a -> H -> MR -> out 1 (reset), b
  // discarded, c passes straight to out 2 (inactive), out 0 prepared.
  {
    Circuit action;
    action.q_in = {"a", "b", "c"};
    action.append(op(Opcode::H, {"a"}, {"a1"}));
    action.append(op(Opcode::MR, {"a1"}, {"a2"}, {"m"}));
    action.append(op(Opcode::DISCARD, {"b"}, {}));
    action.append(op(Opcode::PREP_Z, {}, {"p"}));
    action.q_out = {"p", "a2", "c"};
    action.c_out = {"m"};
    PhysicalInstructionDefinition def("composite", std::move(action));
    std::vector<std::pair<u32, u32>> port_map = {{0, 1}, {2, 2}};
    const auto &in = def.get_q_in();
    const auto &out = def.get_q_out();
    if (def.get_port_map() != port_map || def.get_c_in() != 0 ||
        def.get_c_out() != 1 || in.size() != 3 || out.size() != 3 ||
        in[0].discarded || !in[1].discarded || in[0].inactive ||
        !in[2].inactive || !out[0].prepared || out[1].prepared ||
        !out[1].reset || out[2].reset || !out[2].inactive) {
      std::printf("FAIL: derived signature\n");
      ++failures;
    }

    // A call to that definition follows its port map.
    PhysicalInstructionSet set;
    set.append(std::move(def));
    Circuit outer;
    outer.instruction_set = &set;
    InstructionNode node("call", "composite", {"x", "y", "z"}, {"u", "v", "w"},
                         {}, {"n"});
    if (node.trace(0, outer) != std::optional<std::size_t>(1) ||
        node.trace(1, outer) ||
        node.trace(2, outer) != std::optional<std::size_t>(2)) {
      std::printf("FAIL: trace through an instruction\n");
      ++failures;
    }
  }

  return failures;
}
