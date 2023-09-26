/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

// This is an end-to-end test, so we probably want to put it in a different
// directory.

#include <cmath>
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithm.h>

// Demonstrate NISQ-like sampling for the Phase Estimation algorithm

// Can define this as a free function since it is a pure device quantum kernel
// (cannot be called from host code)
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

// CUDA Quantum Kernel call operators can be templated on input kernel
// expressions. Here we define a general Phase Estimation algorithm that is
// generic on the eigenstate preparation and unitary evolution steps.
struct qpe {

  // Define the call expression to take user-specified eigenstate and unitary
  // evolution kernels, as well as the number of qubits in the counting register
  // and in the eigenstate register.
  template <typename StatePrep, typename Unitary>
  void operator()(const int nCountingQubits, const int nStateQubits,
                  StatePrep &&state_prep, Unitary &&oracle) __qpu__ {
    // Allocate a register of qubits
    cudaq::qreg q(nCountingQubits + nStateQubits);

    // Extract sub-registers, one for the counting qubits another for the
    // eigenstate register
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
      qpe{}, 3, 1, [](cudaq::qspan<> &q) __qpu__ { x(q); }, tgate{});

  // Fine grain access to the bits and counts
  for (auto &[bits, count] : counts) {
    printf("Observed: %s, %lu\n", bits.data(), count);
  }
}

// clang-format off

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_iqft.
// CHECK-SAME:      (%[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant -3.14{{.*}} : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2.0{{.*}} : f64
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_7:.*]] = arith.trunci %[[VAL_6]] : i64 to i32
// CHECK:           %[[VAL_8:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_9:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_12:.*]] = arith.divsi %[[VAL_11]], %[[VAL_4]] : i32
// CHECK:               %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_12]] : i32
// CHECK:               cc.condition %[[VAL_13]]
// CHECK:             } do {
// CHECK:               %[[VAL_14:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:               %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : i32 to i64
// CHECK:               %[[VAL_16:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_15]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_17:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:               %[[VAL_19:.*]] = arith.subi %[[VAL_17]], %[[VAL_18]] : i32
// CHECK:               %[[VAL_20:.*]] = arith.subi %[[VAL_19]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_21:.*]] = arith.extsi %[[VAL_20]] : i32 to i64
// CHECK:               %[[VAL_22:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_21]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.swap %[[VAL_16]], %[[VAL_22]] : (!quake.ref, !quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_23:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_3]] : i32
// CHECK:               cc.store %[[VAL_24]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           cc.scope {
// CHECK:             %[[VAL_25:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:               %[[VAL_27:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_3]] : i32
// CHECK:               %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_28]] : i32
// CHECK:               cc.condition %[[VAL_29]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[VAL_30:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_31:.*]] = arith.extsi %[[VAL_30]] : i32 to i64
// CHECK:                 %[[VAL_32:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_31]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.h %[[VAL_32]] : (!quake.ref) -> ()
// CHECK:                 %[[VAL_33:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                 %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_3]] : i32
// CHECK:                 %[[VAL_35:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[VAL_34]], %[[VAL_35]] : !cc.ptr<i32>
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_36:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:                   %[[VAL_37:.*]] = cc.alloca i32
// CHECK:                   cc.store %[[VAL_36]], %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_38:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_39:.*]] = arith.cmpi sge, %[[VAL_38]], %[[VAL_5]] : i32
// CHECK:                     cc.condition %[[VAL_39]]
// CHECK:                   } do {
// CHECK:                     cc.scope {
// CHECK:                       %[[VAL_40:.*]] = cc.load %[[VAL_35]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_41:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_42:.*]] = arith.subi %[[VAL_40]], %[[VAL_41]] : i32
// CHECK:                       %[[VAL_43:.*]] = math.fpowi %[[VAL_2]], %[[VAL_42]] : f64, i32
// CHECK:                       %[[VAL_44:.*]] = arith.divf %[[VAL_1]], %[[VAL_43]] : f64
// CHECK:                       %[[VAL_45:.*]] = cc.alloca f64
// CHECK:                       cc.store %[[VAL_44]], %[[VAL_45]] : !cc.ptr<f64>
// CHECK:                       %[[VAL_46:.*]] = cc.load %[[VAL_45]] : !cc.ptr<f64>
// CHECK:                       %[[VAL_47:.*]] = cc.load %[[VAL_35]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_48:.*]] = arith.extsi %[[VAL_47]] : i32 to i64
// CHECK:                       %[[VAL_49:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_48]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                       %[[VAL_50:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                       %[[VAL_51:.*]] = arith.extsi %[[VAL_50]] : i32 to i64
// CHECK:                       %[[VAL_52:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_51]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                       quake.r1 (%[[VAL_46]]) {{\[}}%[[VAL_49]]] %[[VAL_52]] : (f64, !quake.ref, !quake.ref) -> ()
// CHECK:                     }
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_53:.*]] = cc.load %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_54:.*]] = arith.subi %[[VAL_53]], %[[VAL_3]] : i32
// CHECK:                     cc.store %[[VAL_54]], %[[VAL_37]] : !cc.ptr<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_55:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:               %[[VAL_56:.*]] = arith.addi %[[VAL_55]], %[[VAL_3]] : i32
// CHECK:               cc.store %[[VAL_56]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_57:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_58:.*]] = arith.subi %[[VAL_57]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_59:.*]] = arith.extsi %[[VAL_58]] : i32 to i64
// CHECK:           %[[VAL_60:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_59]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           quake.h %[[VAL_60]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__tgate(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.t %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_11]] : index
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: index):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: index):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_11]] : index
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_qpeZ4mainE3$_0tgate.
// CHECK-SAME:      (%[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: !cc.callable<(!quake.veq<?>) -> ()>, %[[VAL_3:.*]]: !cc.struct<"tgate" {}>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_10:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : i32 to i64
// CHECK:           %[[VAL_16:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_15]] : i64]
// CHECK:           %[[VAL_17:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = arith.extsi %[[VAL_17]] : i32 to i64
// CHECK:           %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_20:.*]] = quake.subveq %[[VAL_16]], %[[VAL_9]], %[[VAL_19]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           %[[VAL_21:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           %[[VAL_22:.*]] = arith.extsi %[[VAL_21]] : i32 to i64
// CHECK:           %[[VAL_23:.*]] = quake.veq_size %[[VAL_16]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_23]], %[[VAL_8]] : i64
// CHECK:           %[[VAL_25:.*]] = arith.subi %[[VAL_23]], %[[VAL_22]] : i64
// CHECK:           %[[VAL_26:.*]] = quake.subveq %[[VAL_16]], %[[VAL_25]], %[[VAL_24]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// CHECK:           call @__nvqpp__mlirgen__Z4mainE3$_0(%[[VAL_26]]) : (!quake.veq<?>) -> ()
// CHECK:           %[[VAL_27:.*]] = quake.veq_size %[[VAL_20]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_28:.*]] = arith.index_cast %[[VAL_27]] : i64 to index
// CHECK:           %[[VAL_29:.*]] = cc.loop while ((%[[VAL_30:.*]] = %[[VAL_7]]) -> (index)) {
// CHECK:             %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_28]] : index
// CHECK:             cc.condition %[[VAL_31]](%[[VAL_30]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index):
// CHECK:             %[[VAL_33:.*]] = quake.extract_ref %[[VAL_20]]{{\[}}%[[VAL_32]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.h %[[VAL_33]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_32]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_34:.*]]: index):
// CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_6]] : index
// CHECK:             cc.continue %[[VAL_35]] : index
// CHECK:           } {invariant}
// CHECK:           cc.scope {
// CHECK:             %[[VAL_36:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_5]], %[[VAL_36]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_37:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:               %[[VAL_38:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:               %[[VAL_39:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_38]] : i32
// CHECK:               cc.condition %[[VAL_39]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 cc.scope {
// CHECK:                   %[[VAL_40:.*]] = cc.alloca i32
// CHECK:                   cc.store %[[VAL_5]], %[[VAL_40]] : !cc.ptr<i32>
// CHECK:                   cc.loop while {
// CHECK:                     %[[VAL_41:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_42:.*]] = arith.extsi %[[VAL_41]] : i32 to i64
// CHECK:                     %[[VAL_43:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_44:.*]] = arith.extsi %[[VAL_43]] : i32 to i64
// CHECK:                     %[[VAL_45:.*]] = arith.shli %[[VAL_8]], %[[VAL_44]] : i64
// CHECK:                     %[[VAL_46:.*]] = arith.cmpi ult, %[[VAL_42]], %[[VAL_45]] : i64
// CHECK:                     cc.condition %[[VAL_46]]
// CHECK:                   } do {
// CHECK:                     %[[VAL_47:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_48:.*]] = arith.extsi %[[VAL_47]] : i32 to i64
// CHECK:                     %[[VAL_49:.*]] = quake.extract_ref %[[VAL_20]]{{\[}}%[[VAL_48]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     quake.apply @__nvqpp__mlirgen__tgate [%[[VAL_49]]] %[[VAL_26]] : (!quake.ref, !quake.veq<?>) -> ()
// CHECK:                     cc.continue
// CHECK:                   } step {
// CHECK:                     %[[VAL_50:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i32>
// CHECK:                     %[[VAL_51:.*]] = arith.addi %[[VAL_50]], %[[VAL_4]] : i32
// CHECK:                     cc.store %[[VAL_51]], %[[VAL_40]] : !cc.ptr<i32>
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_52:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_52]], %[[VAL_4]] : i32
// CHECK:               cc.store %[[VAL_53]], %[[VAL_36]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           call @__nvqpp__mlirgen__function_iqft{{.*}}(%[[VAL_20]]) : (!quake.veq<?>) -> ()
// CHECK:           %[[VAL_54:.*]] = quake.mz %[[VAL_20]] : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }

