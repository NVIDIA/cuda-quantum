/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <iomanip>
#include <iostream>

struct bellCircuit {
  void operator()(const std::vector<double> vec) __qpu__ {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    cx(qubits[0], qubits[1]);
  }
};

struct noOpCircuit {
  void operator()(const std::vector<double> vec) __qpu__ {
    cudaq::qvector qubits(2);
  }
};

struct trotter {
  void operator()(cudaq::state *initial_state,
                  std::vector<double> coefficients) __qpu__ {
    cudaq::qvector q(initial_state);
  }
};

int main() {
  std::vector<double> a(10, -0.6);
  std::vector<double> b(10, 0.3);
  auto state1 = cudaq::get_state(bellCircuit{}, a);
  auto state2 = cudaq::get_state(noOpCircuit{}, b);
  const auto overlap = state1.overlap(state2);
  std::cout << std::fixed << std::setprecision(1);
  std::cout << "Overlap is " << overlap << std::endl;
  // CHECK: Overlap is (0.7,0.0)

  auto state3 = cudaq::get_state(trotter{}, &state1, a);
  const auto overlap2 = state1.overlap(state3);
  std::cout << "Overlap is " << overlap2 << std::endl;
  // CHECK: Overlap is (1.0,0.0)
  return 0;
}
