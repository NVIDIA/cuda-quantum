/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum %s -o %t && %t || echo "passed" |& FileCheck %s

#include <cudaq.h>
#include <cstdio>

struct Qernel_A {
  void operator()(std::vector<double> angles,
                  std::vector<cudaq::pauli_word> paulis) __qpu__ {
    cudaq::qvector q(3);
    for (int i = 0; i < angles.size(); i++) {
      exp_pauli(angles[i], q, paulis[i]);
    }
  }
};

int main() {
  Qernel_A a;
  std::vector<double> v = {1.0, 2.0};
  std::vector<cudaq::pauli_word> p = {"XYZ", "IXX"};
  printf("calling sample\n");
  cudaq::sample(a, v, p);
  return 0;
}

// CHECK-NOT: cannot determine pauli word string
