/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Regression test for https://github.com/NVIDIA/cuda-quantum/issues/4350
// (`createStateFromData` was called via `CreateStateOp` in the middle of kernel
// execution, after qubits were already live in the simulator, corrupting the
// active state).

// Compile and execute; verifies the runtime produces the correct bitstrings.
// RUN: nvq++ %s -o %t && %t | FileCheck %s

// Lower to Quake IR and verify the expected IR structure.
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s --check-prefix=MLIR

#include <cudaq.h>
#include <iostream>
#include <vector>

// Allocate one qubit first, then allocate more qubits from a state vector.
// The second qalloc emits `quake.create_state` which calls
// `createStateFromData` while the simulator already owns the live state for
// `p`.
__qpu__ void test_single_then_state(std::vector<cudaq::complex> inState) {
  cudaq::qubit p;
  cudaq::qvector q{cudaq::state(inState)};
  mz(p);
  mz(q);
}

// Same pattern with a multi-qubit register allocated first.
__qpu__ void test_multi_then_state(std::vector<cudaq::complex> inState) {
  cudaq::qvector p(2);
  cudaq::qvector q{cudaq::state(inState)};
  mz(p);
  mz(q);
}

int main() {
  constexpr int shots = 1000;
  // State vector for |11> (2 qubits, 4 amplitudes, last = 1.0).
  std::vector<cudaq::complex> state2q{0., 0., 0., 1.};

  {
    // p starts in |0>; q is initialized to |11>.
    // All shots must produce "011".
    auto counts = cudaq::sample(shots, test_single_then_state, state2q);
    std::cout << "single qubit + state: ";
    counts.dump();
    if (counts.count("011") != static_cast<std::size_t>(shots)) {
      std::cerr << "FAIL: expected all " << shots << " shots as '011', got "
                << counts.count("011") << "\n";
      return 1;
    }
    std::cout << "single_then_state PASSED\n";
  }

  {
    // p[0],p[1] start in |00>; q is initialized to |11>.
    // All shots must produce "0011".
    auto counts = cudaq::sample(shots, test_multi_then_state, state2q);
    std::cout << "multi-qubit + state: ";
    counts.dump();
    if (counts.count("0011") != static_cast<std::size_t>(shots)) {
      std::cerr << "FAIL: expected all " << shots << " shots as '0011', got "
                << counts.count("0011") << "\n";
      return 1;
    }
    std::cout << "multi_then_state PASSED\n";
  }

  return 0;
}

// CHECK: single_then_state PASSED
// CHECK: multi_then_state PASSED

// Verify the mid-execution state-init IR: `quake.alloca` for the prior
// qubits appears before `quake.create_state`, i.e., `createStateFromData` call
// while the simulator already owns live qubits.

// The mangled suffix after the dot is compiler-dependent (GCC vs Clang libc++),
// so the LABEL stops at the common prefix.
// clang-format off
// MLIR-LABEL: func.func @__nvqpp__mlirgen__function_test_single_then_state.
// MLIR:         %[[VAL_1:.*]] = quake.alloca !quake.ref
// MLIR:         %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0:.*]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
// MLIR:         %[[VAL_3:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<complex<f64>>) -> i64
// MLIR:         %[[VAL_4:.*]] = quake.create_state %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<complex<f64>>, i64) -> !cc.ptr<!quake.state>
// MLIR:         %[[VAL_5:.*]] = quake.get_number_of_qubits %[[VAL_4]] : (!cc.ptr<!quake.state>) -> i64
// MLIR:         %[[VAL_6:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_5]] : i64]
// MLIR:         %[[VAL_7:.*]] = quake.init_state %[[VAL_6]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// MLIR:         quake.delete_state %[[VAL_4]] : !cc.ptr<!quake.state>
// MLIR:         %[[VAL_8:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// MLIR:         %[[VAL_9:.*]] = quake.mz %[[VAL_7]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// MLIR:         return
// MLIR:       }

// MLIR-LABEL: func.func @__nvqpp__mlirgen__function_test_multi_then_state.
// MLIR:         %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// MLIR:         %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0:.*]] : (!cc.stdvec<complex<f64>>) -> !cc.ptr<complex<f64>>
// MLIR:         %[[VAL_3:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<complex<f64>>) -> i64
// MLIR:         %[[VAL_4:.*]] = quake.create_state %[[VAL_2]], %[[VAL_3]] : (!cc.ptr<complex<f64>>, i64) -> !cc.ptr<!quake.state>
// MLIR:         %[[VAL_5:.*]] = quake.get_number_of_qubits %[[VAL_4]] : (!cc.ptr<!quake.state>) -> i64
// MLIR:         %[[VAL_6:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_5]] : i64]
// MLIR:         %[[VAL_7:.*]] = quake.init_state %[[VAL_6]], %[[VAL_4]] : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
// MLIR:         quake.delete_state %[[VAL_4]] : !cc.ptr<!quake.state>
// MLIR:         %[[VAL_8:.*]] = quake.mz %[[VAL_1]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// MLIR:         %[[VAL_9:.*]] = quake.mz %[[VAL_7]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// MLIR:         return
// MLIR:       }
// clang-format on
