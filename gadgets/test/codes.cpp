/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#include "cudaq/gadgets/codes/subsystem_code.h"

#include <cstdio>
#include <stdexcept>

using namespace cudaq::gadgets;
using P = Pauli;

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

SparsePauliString zz(std::size_t a, std::size_t b) {
  return {false, {{P::Z, a}, {P::Z, b}}};
}
} // namespace

SparsePauliString x(std::vector<std::size_t> qubits) {
  SparsePauliString pauli;
  for (std::size_t q : qubits)
    pauli.ops.push_back({P::X, q});
  return pauli;
}

int main() {
  // Repetition code: generators Z0Z1, Z1Z2, and the redundant Z0Z2.
  SubsystemCode rep3("rep3", 3, {zz(0, 1), zz(1, 2), zz(0, 2)}, {0, 1},
                     {{x({0, 1, 2}), {false, {{P::Z, 0}}}}});
  if (rep3.get_name() != "rep3" || rep3.get_num_qubits() != 3 ||
      rep3.get_generators().size() != 3 || rep3.get_logicals().size() != 1 ||
      !rep3.get_gauges().empty()) {
    std::printf("FAIL: repetition code\n");
    ++failures;
  }

  // Toy code: stabilizer Z0Z1 and one gauge qubit on two physical qubits.
  SubsystemCode gauge("gauge", 2, {zz(0, 1)}, {0}, {},
                      {{x({0, 1}), {false, {{P::Z, 0}}}}});
  if (gauge.get_gauges().size() != 1) {
    std::printf("FAIL: gauge code\n");
    ++failures;
  }

  expect_throw("independent index out of range",
               [] { SubsystemCode("bad", 2, {zz(0, 1)}, {0, 1}, {}); });
  expect_throw("qubit outside the code", [] {
    SubsystemCode("bad", 2, {zz(0, 2)}, {0}, {{x({0}), x({1})}});
  });
  expect_throw("counts do not add up", [] {
    SubsystemCode("bad", 3, {zz(0, 1)}, {0}, {{x({0}), x({1})}});
  });

  return failures;
}
