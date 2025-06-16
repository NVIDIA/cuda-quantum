/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ types_qubit_example.cpp && ./a.out`

#include <cudaq.h>
#include <stdio.h> // For printf

// Kernel to demonstrate qubit allocation and ID management
__qpu__ void qubit_usage_kernel_cpp() {
  // [Begin CppQubitUsage]
  {
    // Allocate a qubit in the |0> state
    cudaq::qubit q;
    // Put the qubit in a superposition of |0> and |1>
    h(q);                         // cudaq::h == hadamard, ADL leveraged
    printf("ID = %lu\n", q.id()); // prints 0
    cudaq::qubit r;
    printf("ID = %lu\n", r.id()); // prints 1
    // qubit out of scope, implicit deallocation
    mz(q); // Add measurements for sample
    mz(r);
  }
  cudaq::qubit q_realloc; // Renamed to avoid conflict in broader scope if this
                          // was a real test
  printf("ID = %lu\n", q_realloc.id()); // prints 0 (previous deallocated)
  mz(q_realloc);
  // [End CppQubitUsage]
}

int main() {
  printf("C++ Qubit Usage Example:\n");
  auto counts = cudaq::sample(qubit_usage_kernel_cpp);
  // Due to printf, actual counts might be less critical here for the test's
  // main purpose. The printf calls demonstrate ID management.
  counts.dump();
  return 0;
}

