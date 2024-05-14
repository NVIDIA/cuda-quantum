/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std %s -o %t && %t | FileCheck %s

#include <cudaq.h>

// __qpu__ void test(std::vector<cudaq::complex> inState) {
//   cudaq::qvector q = inState;
// }

// __qpu__ void test(std::vector<std::complex<double>> inState) {
//   cudaq::qvector q = inState;
// }

// __qpu__ void test(std::vector<double>& inState) {
//   std::vector<cudaq::complex> ret(inState.size());
//   int i = 0;
//   for (auto& x : inState) {
//     ret[i++] = x;
//   }
//   cudaq::qvector q = ret;
// }

// CHECK: size 2

// int main() {
//   std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
//   auto counts = cudaq::sample(test, vec);
//   counts.dump();

//   printf("size %zu\n", counts.size());
// }

__qpu__ void test(std::vector<cudaq::complex> inState) {
  cudaq::qvector q = inState;

  // Can now operate on the qvector as usual:
  // Rotate state of the front qubit 180 degrees along X.
  x(q.front());
  // Rotate state of the back qubit 180 degrees along Y.
  y(q.back());
  // Put qubits into superposition state.
  h(q);

  // Measure.
  mz(q);
}


int main() {
  std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
  auto counts = cudaq::sample(test, vec);
  counts.dump();

  printf("size %zu\n", counts.size());


}