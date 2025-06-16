/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ kernels_composition_templates.cpp && ./a.out`

#include <cudaq.h>
#include <functional>
#include <stdio.h>

// [Begin Kernel Composition Templates C++]
__qpu__ void MyStatePrep_cpp(cudaq::qview<> qubits) {
  if (!qubits.empty()) {
    h(qubits[0]);
  }
  printf("MyStatePrep_cpp executed.\n");
}

struct MyGenericAlgorithm_cpp {
  template <typename StatePrep>
  void operator()(const StatePrep &statePrep) __qpu__ {
    cudaq::qarray<10> q; // Example size
    statePrep(q);
    // ... other algorithm steps ...
    mz(q); // Measure to make it a full circuit
    printf("MyGenericAlgorithm_cpp executed.\n");
  }
};

// -or- with placeholder type specifiers
struct MyGenericAlgorithm2_cpp {
  void operator()(const auto &statePrep) __qpu__ {
    cudaq::qarray<10> q; // Example size
    statePrep(q);
    // ... other algorithm steps ...
    mz(q); // Measure
    printf("MyGenericAlgorithm2_cpp executed.\n");
  }
};

// Host function to call the entry-point kernels
void callKernels_cpp() {
  MyGenericAlgorithm_cpp algorithm;
  // MyStatePrep_cpp is __qpu__ only, cannot be passed directly as callable from
  // host to __qpu__ operator() The generic algorithm itself must be an entry
  // point, or called by one. Let's make an entry point that uses
  // MyGenericAlgorithm_cpp
  auto entry_point_algo1 = [&]() __qpu__ { algorithm(MyStatePrep_cpp); };
  cudaq::sample(entry_point_algo1);

  MyGenericAlgorithm2_cpp anotherVersion;
  auto entry_point_algo2 = [&]() __qpu__ { anotherVersion(MyStatePrep_cpp); };
  cudaq::sample(entry_point_algo2);
}
// [End Kernel Composition Templates C++]

int main() {
  // [Begin Kernel Composition Templates C++ Execution]
  callKernels_cpp();
  // [End Kernel Composition Templates C++ Execution]
  return 0;
}

