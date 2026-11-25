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

Node op(Opcode opcode, Wires q_in, Wires q_out, Wires c_out = {}) {
  return Node("", opcode, std::move(q_in), std::move(q_out), {},
              std::move(c_out));
}

/// Appends node to an empty semantic circuit.
void append(Node node) { Circuit().append(std::move(node)); }

/// H a; CX a b; MEASURE_Z a -> m; X b if m.
Circuit bell_correction() {
  Circuit circuit({"a0", "b0"}, {"b2"}, {}, {"m"});
  circuit.append(op(Opcode::H, {"a0"}, {"a1"}));
  circuit.append(op(Opcode::CX, {"a1", "b0"}, {"a2", "b1"}));
  auto [id, outcomes] =
      circuit.append(Node("measure", Opcode::MEASURE_Z, {"a2"}, {}, {}, {"m"}));
  if (id != "measure" || outcomes != Wires{"m"}) {
    std::printf("FAIL: append result\n");
    ++failures;
  }
  circuit.append(Node("fix", Opcode::CONDITIONAL_PAULI, {"b1"}, {"b2"}, {"m"},
                      {}, ConditionalPauli{PauliString::from_string("+X"), 1}));
  return circuit;
}
} // namespace

int main() {
  expect_ok("valid circuit", [] { bell_correction().validate(); });
  expect_ok("random bit, idle, and reset", [] {
    Circuit circuit({"a"}, {"c"}, {}, {"r"});
    circuit.append(Node("random", Opcode::RANDOM_BIT, {}, {}, {}, {"r"}));
    circuit.append(op(Opcode::IDLE, {"a"}, {"b"}));
    circuit.append(op(Opcode::RESET, {"b"}, {"c"}));
    circuit.validate();
  });

  expect_throw("wire counts", [] { append(op(Opcode::CX, {"a"}, {"b"})); });
  expect_throw("unknown opcode", [] {
    append(Node("x", std::size_t(num_opcodes), {"a"}, {"b"}));
  });
  expect_throw("conditional Pauli width", [] {
    append(Node("fix", Opcode::CONDITIONAL_PAULI, {"a"}, {"b"}, {}, {},
                ConditionalPauli{PauliString::from_string("+ZZ")}));
  });
  expect_throw("conditional Pauli bits", [] {
    append(Node("fix", Opcode::CONDITIONAL_PAULI, {"a"}, {"b"}, {"m", "n"}, {},
                ConditionalPauli{PauliString::from_string("+Z"), 1}));
  });
  expect_throw("Clifford images", [] {
    CliffordImages({PauliString::from_string("X")},
                   {PauliString::from_string("ZZ")});
  });
  expect_throw("non-Hermitian Pauli", [] {
    append(Node("m", Opcode::PAULI_MEASUREMENT, {"a"}, {"b"}, {}, {"m"},
                PauliString::from_string("+iZ")));
  });
  expect_throw("missing param", [] { append(op(Opcode::RX, {"a"}, {"b"})); });
  expect_throw("unexpected payload", [] {
    append(Node("x", Opcode::X, {"a"}, {"b"}, {}, {}, ParamValue(1.0)));
  });
  expect_throw("unitary size", [] {
    append(Node("u", Opcode::UNITARY, {"a"}, {"b"}, {}, {},
                Unitary(std::vector<std::complex<double>>(2))));
  });
  expect_ok("payloads", [] {
    append(Node("rx", Opcode::RX, {"a"}, {"b"}, {}, {}, ParamValue("theta")));
    append(Node(
        "swap", Opcode::CLIFFORD, {"a", "b"}, {"c", "d"}, {}, {},
        CliffordImages{
            {PauliString::from_string("IX"), PauliString::from_string("XI")},
            {PauliString::from_string("IZ"), PauliString::from_string("ZI")}}));
    append(Node("mzz", Opcode::PAULI_MEASUREMENT, {"a", "b"}, {"c", "d"}, {},
                {"m0", "m1"}, PauliString::from_string("ZZ")));
    append(Node("s", Opcode::DIAGONAL_UNITARY, {"a"}, {"b"}, {}, {},
                DiagonalUnitary({1, {0, 1}})));
    append(Node("cz", Opcode::CONDITIONAL_PAULI, {"a"}, {"b"}, {"m", "n"}, {},
                ConditionalPauli{PauliString::from_string("+Z"), 2, true}));
  });

  PhysicalInstructionSet native;
  Circuit cz({"a", "b"}, {"c", "d"});
  cz.append(op(Opcode::CZ, {"a", "b"}, {"c", "d"}));
  native.append(PhysicalInstructionDefinition("CZ", std::move(cz)));
  std::size_t cz_index = *native.index_of("CZ");
  auto physical = [&] {
    return Circuit({"a", "b"}, {"c", "d"}, {}, {}, &native);
  };
  expect_ok("instruction", [&] {
    Circuit circuit = physical();
    circuit.append(Node("cz", cz_index, {"a", "b"}, {"c", "d"}));
    circuit.validate();
  });
  expect_throw("instruction wire count",
               [&] { physical().append(Node("cz", cz_index, {"a"}, {"c"})); });
  expect_throw("instruction not in the set", [&] {
    physical().append(Node("cz", cz_index + 1, {"a", "b"}, {"c", "d"}));
  });
  expect_throw("payload on a physical instruction", [&] {
    physical().append(
        Node("cz", cz_index, {"a", "b"}, {"c", "d"}, {}, {}, ParamValue(1.0)));
  });
  expect_throw("action over an instruction set", [&] {
    Circuit action = physical();
    action.append(Node("cz", cz_index, {"a", "b"}, {"c", "d"}));
    PhysicalInstructionDefinition("wrapped", std::move(action));
  });

  // Signature derived by tracing: a -> H -> RESET -> out 1, b measured
  // destructively, c straight to out 2, out 0 prepared.
  {
    Circuit action({"a", "b", "c"}, {"p", "a2", "c"}, {}, {"m"});
    action.append(op(Opcode::H, {"a"}, {"a1"}));
    action.append(op(Opcode::RESET, {"a1"}, {"a2"}));
    action.append(op(Opcode::MEASURE_Z, {"b"}, {}, {"m"}));
    action.append(op(Opcode::PREPARE_Z, {}, {"p"}));
    PhysicalInstructionDefinition def("composite", std::move(action));
    std::vector<std::pair<std::size_t, std::size_t>> port_map = {{0, 1},
                                                                 {2, 2}};
    if (def.get_port_map() != port_map ||
        def.get_shape() != Shape{3, 3, 0, 1}) {
      std::printf("FAIL: derived signature\n");
      ++failures;
    }
  }

  return failures;
}
