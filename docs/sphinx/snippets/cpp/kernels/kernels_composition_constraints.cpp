/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ kernels_composition_constraints.cpp && ./a.out`

#include <cudaq.h>
#include <functional>
#include <stdio.h>
#include <type_traits>
#include <vector>

// [Begin Kernel Composition Constraints C++]
namespace cudaq_constraints_example { // Renamed to avoid conflict if cudaq
                                      // namespace has similar
// Generic constraint on Kernel Function Signatures
template <typename Kernel, typename Signature>
concept signature_concept =
    std::is_convertible_v<Kernel, std::function<Signature>>;

// Specialized for taking a single qubit reference
template <typename Kernel>
concept takes_qubit_ref_concept =
    signature_concept<Kernel, void(cudaq::qubit &)>;

// Specialized for taking a qarray reference
template <typename Kernel>
concept takes_qarray_ref_concept =
    signature_concept<Kernel, void(cudaq::qarray<10> &)>; // Example fixed size
} // namespace cudaq_constraints_example

// Example kernels satisfying constraints
__qpu__ void state_prep_on_qarray_example(cudaq::qarray<10> &q) {
  h(q[0]);
  printf("state_prep_on_qarray_example executed.\n");
}

__qpu__ void state_prep_on_qubit_example(cudaq::qubit &q_ref) {
  h(q_ref);
  printf("state_prep_on_qubit_example executed.\n");
}

struct MyGenericAlgorithmOnQarray_cpp {
  // Using cudaq::qarray<10>& directly in concept for this example
  void operator()(cudaq_constraints_example::takes_qarray_ref_concept auto
                      &&statePrep) __qpu__ {
    cudaq::qarray<10> q_local; // Kernel defines its own qarray
    statePrep(q_local);        // Pass it to the state prep kernel
    // ... other algorithm steps ...
    mz(q_local);
    printf("MyGenericAlgorithmOnQarray_cpp executed.\n");
  }
};

struct MyGenericAlgorithmOnQubit_cpp {
  void operator()(cudaq_constraints_example::takes_qubit_ref_concept auto
                      &&statePrep) __qpu__ {
    cudaq::qarray<10> q_local; // Kernel defines its qarray
    statePrep(q_local[0]);     // Pass a single qubit from it
    // ... other algorithm steps ...
    mz(q_local);
    printf("MyGenericAlgorithmOnQubit_cpp executed.\n");
  }
};

// Host function to call the entry-point kernels
void callConstrainedKernels_cpp() {
  MyGenericAlgorithmOnQarray_cpp algo_qarray;
  auto entry_algo_qarray = [&]() __qpu__ {
    algo_qarray(state_prep_on_qarray_example);
  };
  cudaq::sample(entry_algo_qarray);

  MyGenericAlgorithmOnQubit_cpp algo_qubit;
  auto entry_algo_qubit = [&]() __qpu__ {
    algo_qubit(state_prep_on_qubit_example);
  };
  cudaq::sample(entry_algo_qubit);
}
// [End Kernel Composition Constraints C++]
// [Begin Kernel Composition Constraints C++ Execution]
int main() {
  callConstrainedKernels_cpp();
  return 0;
}
// [End Kernel Composition Constraints C++ Execution]

