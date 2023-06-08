/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// This is an end-to-end test, so we probably want to put it in a different
// directory.

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <stdio.h>

#include <cmath>

// Demonstrate NISQ-like sampling for the Phase Estimation algorithm

// Can define this as a free function since it is
// a pure device quantum kernel (cannot be called from host code)
__qpu__ void iqft(cudaq::qspan<> q) {
  int N = q.size();
  // Swap qubits
  for (int i = 0; i < N / 2; ++i) {
    swap(q[i], q[N - i - 1]);
  }

  for (int i = 0; i < N - 1; ++i) {
    h(q[i]);
    int j = i + 1;
    for (int y = i; y >= 0; --y) {
      const double theta = -M_PI / std::pow(2.0, j - y);
      r1<cudaq::ctrl>(theta, q[j], q[y]);
    }
  }

  h(q[N - 1]);
}

// Define an oracle CUDA Quantum kernel
struct tgate {
  // We do not own the qubits here, so just use a qspan.
  void operator()(cudaq::qspan<> &q) __qpu__ { t(q); }
};

// CUDA Quantum Kernel call operators can be templated on
// input kernel expressions. Here we define a general
// Phase Estimation algorithm that is generic on the eigenstate
// preparation and unitary evolution steps.
struct qpe {

  // Define the call expression to take user-specified eigenstate
  // and unitary evolution kernels, as well as the number of qubits in the
  // counting register and in the eigenstate register.
  template <typename StatePrep, typename Unitary>
  void operator()(const int nCountingQubits, const int nStateQubits ,
                  StatePrep &&state_prep, Unitary &&oracle) __qpu__ {
    // Allocate a register of qubits
    cudaq::qreg q(nCountingQubits + nStateQubits);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(nCountingQubits);
    auto state_register = q.back(nStateQubits);

    // Prepare the eigenstate
    state_prep(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform ctrl-U^j
    for (int i = 0; i < nCountingQubits; ++i) {
      for (int j = 0; j < (1UL << i); ++j) {
        cudaq::control(oracle, {counting_qubits[i]}, state_register);
      }
    }

    // Apply inverse quantum fourier transform
    iqft(counting_qubits);

    // Measure to gather sampling statistics
    mz(counting_qubits);

    return;
  }
};

int main() {
  // Sample the QPE kernel for 3 counting qubits, 1 state qubit, a |1>
  // eigenstate preparation kernel, and a T gate unitary.
  auto counts = cudaq::sample(
      qpe{}, 3, 1 , [](cudaq::qspan<> &q) __qpu__ { x(q); }, tgate{});

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_iqft
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_2:.*]] = arith.trunci %[[VAL_1]] : i64 to i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_5:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_7:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_8:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_9:.*]] = arith.divsi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:               %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_9]] : i32
// CHECK:               cc.condition %[[VAL_10]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_11:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_12:.*]] = arith.extsi %[[VAL_11]] : i32 to i64
// CHECK:                 %[[VAL_13:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 %[[VAL_14:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_15:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_16:.*]] = arith.subi %[[VAL_14]], %[[VAL_15]] : i32
// CHECK:                 %[[VAL_17:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_18:.*]] = arith.subi %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:                 %[[VAL_19:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:                 %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.swap %[[VAL_13]], %[[VAL_20]] : (
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_21:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:               %[[VAL_22:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_22]] : i32
// CHECK:               cc.store %[[VAL_23]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           cc.scope {
// CHECK:             %[[VAL_24:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_25:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_24]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:               %[[VAL_27:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:               %[[VAL_28:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_29:.*]] = arith.subi %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_29]] : i32
// CHECK:               cc.condition %[[VAL_30]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_31:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_32:.*]] = arith.extsi %[[VAL_31]] : i32 to i64
// CHECK:                 %[[VAL_33:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_32]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.h %[[VAL_33]] : (!quake.ref) -> ()
// CHECK:                 %[[VAL_34:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_35:.*]] = arith.constant 1 : i32
// CHECK:                 %[[VAL_36:.*]] = arith.addi %[[VAL_34]], %[[VAL_35]] : i32
// CHECK:                 %[[VAL_37:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[VAL_36]], %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_38:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                   %[[VAL_39:.*]] = cc.alloca i32
// CHECK:                   cc.store %[[VAL_38]], %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_40:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_41:.*]] = arith.constant 0 : i32
// CHECK:                     %[[VAL_42:.*]] = arith.cmpi sge, %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:                     cc.condition %[[VAL_42]]
// CHECK:                   } do {
// CHECK:                     cc.scope {
// CHECK:                       %[[VAL_43:.*]] = arith.constant 3.1415926535897931 : f64
// CHECK:                       %[[VAL_44:.*]] = arith.constant -1.000000e+00 : f64
// CHECK:                       %[[VAL_45:.*]] = arith.mulf %[[VAL_43]], %[[VAL_44]] : f64
// CHECK:                       %[[VAL_46:.*]] = arith.constant 2.000000e+00 : f64
// CHECK:                       %[[VAL_47:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_48:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_49:.*]] = arith.subi %[[VAL_47]], %[[VAL_48]] : i32
// CHECK:                       %[[VAL_51:.*]] = math.fpowi %[[VAL_46]], %[[VAL_49]] : f64, i32
// CHECK:                       %[[VAL_52:.*]] = arith.divf %[[VAL_45]], %[[VAL_51]] : f64
// CHECK:                       %[[VAL_53:.*]] = cc.alloca f64
// CHECK:                       cc.store %[[VAL_52]], %[[VAL_53]] : !cc.ptr<f64>
// CHECK:                       %[[VAL_54:.*]] = cc.load %[[VAL_53]] : !cc.ptr<f64>
// CHECK:                       %[[VAL_55:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_56:.*]] = arith.extsi %[[VAL_55]] : i32 to i64
// CHECK:                       %[[VAL_57:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_56]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                       %[[VAL_58:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_59:.*]] = arith.extsi %[[VAL_58]] : i32 to i64
// CHECK:                       %[[VAL_60:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_59]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                       quake.r1 (%[[VAL_54]]) [%[[VAL_57]]] %[[VAL_60]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:                     }
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_61:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_62:.*]] = arith.constant 1 : i32
// CHECK:                     %[[VAL_63:.*]] = arith.subi %[[VAL_61]], %[[VAL_62]] : i32
// CHECK:                     cc.store %[[VAL_63]], %[[VAL_39]] : !cc.ptr<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_64:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:               %[[VAL_65:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[VAL_64]], %[[VAL_65]] : i32
// CHECK:               cc.store %[[VAL_66]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_67:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_68:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_69:.*]] = arith.subi %[[VAL_67]], %[[VAL_68]] : i32
// CHECK:           %[[VAL_70:.*]] = arith.extsi %[[VAL_69]] : i32 to i64
// CHECK:           %[[VAL_71:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_70]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           quake.h %[[VAL_71]]
// CHECK:           return
// CHECK:         }


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__tgate
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.veq<?>) attributes
// CHECK:           %[[VAL_3:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_1]]) -> (index)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.t %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_11]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.veq<?>)
// CHECK:           %[[VAL_3:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_1]]) -> (index)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.x %[[VAL_9]] :
// CHECK:             cc.continue %[[VAL_8]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_11]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_qpe
// CHECK-SAME:      (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32,
// CHECK-SAME:       %[[VAL_2:.*]]: !cc.lambda<(!quake.veq<?>) -> ()>,
// CHECK-SAME:       %[[VAL_3:.*]]: !cc.struct<"tgate" {}>) attributes
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_6]], %[[VAL_7]] : i32
// CHECK:           %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : i32 to i64
// CHECK:           %[[VAL_10:.*]] = quake.alloca !quake.veq<?>[%[[VAL_9]] : i64]
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = arith.extsi %[[VAL_11]] : i32 to i64
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_15:.*]] = arith.subi %[[VAL_12]], %[[VAL_14]] : i64
// CHECK:           %[[VAL_16:.*]] = quake.subvec %[[VAL_10]], %[[VAL_13]], %[[VAL_15]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_17:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : i32 to i64
// CHECK:           %[[VAL_19:.*]] = quake.vec_size %[[VAL_10]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_20:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_21:.*]] = arith.subi %[[VAL_19]], %[[VAL_20]] : i64
// CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_19]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_23:.*]] = quake.subvec %[[VAL_10]], %[[VAL_22]], %[[VAL_21]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           call @__nvqpp__mlirgen__Z4mainE3$_0(%[[VAL_23]]) : (!quake.veq<?>) -> ()
// CHECK:           %[[VAL_24:.*]] = quake.vec_size %[[VAL_16]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_25:.*]] = arith.index_cast %[[VAL_24]] : i64 to index
// CHECK:           %[[VAL_26:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_27:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_28:.*]] = cc.loop while ((%[[VAL_29:.*]] = %[[VAL_26]]) -> (index)) {
// CHECK:             %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_29]], %[[VAL_25]] : index
// CHECK:             cc.condition %[[VAL_30]](%[[VAL_29]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_31:.*]]: index):
// CHECK:             %[[VAL_32:.*]] = quake.extract_ref %[[VAL_16]]{{\[}}%[[VAL_31]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.h %[[VAL_32]] :
// CHECK:             cc.continue %[[VAL_31]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_33:.*]]: index):
// CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_27]] : index
// CHECK:             cc.continue %[[VAL_34]] : index
// CHECK:           }
// CHECK:           cc.scope {
// CHECK:             %[[VAL_35:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_36:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_35]], %[[VAL_36]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_37:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:               %[[VAL_38:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_39:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               cc.condition %[[VAL_39]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_40:.*]] = arith.constant 0 : i32
// CHECK:                   %[[VAL_41:.*]] = cc.alloca i32
// CHECK:                   cc.store %[[VAL_40]], %[[VAL_41]] : !cc.ptr<i32>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_42:.*]] = cc.load %[[VAL_41]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_43:.*]] = arith.extsi %[[VAL_42]] : i32 to i64
// CHECK:                     %[[VAL_44:.*]] = arith.constant 1 : i64
// CHECK:                     %[[VAL_45:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_46:.*]] = arith.extsi %[[VAL_45]] : i32 to i64
// CHECK:                     %[[VAL_47:.*]] = arith.shli %[[VAL_44]], %[[VAL_46]] : i64
// CHECK:                     %[[VAL_48:.*]] = arith.cmpi ult, %[[VAL_43]], %[[VAL_47]] : i64
// CHECK:                     cc.condition %[[VAL_48]]
// CHECK:                   } do {
// CHECK:                     cc.scope {
// CHECK:                       %[[VAL_49:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_50:.*]] = arith.extsi %[[VAL_49]] : i32 to i64
// CHECK:                       %[[VAL_51:.*]] = quake.extract_ref %[[VAL_16]]{{\[}}%[[VAL_50]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                       quake.apply @__nvqpp__mlirgen__tgate[%[[VAL_51]]] %[[VAL_23]] : (!quake.ref, !quake.veq<?>) -> ()
// CHECK:                     }
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_52:.*]] = cc.load %[[VAL_41]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_53:.*]] = arith.constant 1 : i32
// CHECK:                     %[[VAL_54:.*]] = arith.addi %[[VAL_52]], %[[VAL_53]] : i32
// CHECK:                     cc.store %[[VAL_54]], %[[VAL_41]] : !cc.ptr<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_55:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:               %[[VAL_56:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_57:.*]] = arith.addi %[[VAL_55]], %[[VAL_56]] : i32
// CHECK:               cc.store %[[VAL_57]], %[[VAL_36]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           call @__nvqpp__mlirgen__function_iqft{{.*}}(%[[VAL_16]]) : (!quake.veq<?>) -> ()
// CHECK:           %[[VAL_68:.*]] = quake.mz %[[VAL_16]] : (!quake.veq<?>) ->  !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }
