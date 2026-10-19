/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ first_observe.cpp && ./a.out`
// [Begin Observe1]
#include <cudaq.h>
#include <cudaq/algorithm.h>

#include <iostream>

__qpu__ void kernel() {
  cudaq::qubit qubit;
  h(qubit);
}

int main() {
  auto spin_operator = cudaq::spin_op::z(0);
  std::cout << spin_operator.to_string() << "\n";
  // [End Observe1]

  // [Begin Observe2]
  auto result_0 = cudaq::observe(kernel, spin_operator);
  // Expectation value of kernel with respect to single `Z` term
  // should print: 0.0
  std::cout << "<kernel | spin_operator | kernel> = " << result_0.expectation()
            << "\n";
  // [End Observe2]

  // [Begin Observe3]
  auto result_1 = cudaq::observe(1000, kernel, spin_operator);
  // Expectation value of kernel with respect to single `Z` term,
  // but instead of a single deterministic execution of the kernel,
  // we sample over 1000 shots. We should now print an expectation
  // value that is close to, but not quite, zero.
  // Example: 0.025
  std::cout << "<kernel | spin_operator | kernel> = " << result_1.expectation()
            << "\n";
}
// [End Observe3]
