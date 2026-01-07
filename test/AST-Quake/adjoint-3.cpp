/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --apply-op-specialization | FileCheck --check-prefix=ADJOINT %s

#include <cudaq.h>

struct statePrep_A {
  void operator()(cudaq::qvector<> &q, const double bmax) __qpu__ {

    int n = q.size();
    // all qubits sans ancilla
    auto qubit_subset = q.front(n - 1);

    h(qubit_subset);

    ry(bmax / pow(2.0, n - 1), q[n - 1]);

    for (int i = 1; i < n; i++) {
      ry<cudaq::ctrl>(bmax / pow(2.0, n - i - 1), q[i - 1], q[n - 1]);
    }
  }
};

struct QernelZero {
  void operator()(cudaq::qvector<> &q) __qpu__ {

    auto ctrl_qubits = q.front(q.size() - 1);
    auto &last_qubit = q.back();

    x(q);
    h(last_qubit);
    x<cudaq::ctrl>(ctrl_qubits, last_qubit);
    h(last_qubit);
    x(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__statePrep_A
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__QernelZero

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__run_circuit
// CHECK-SAME:        (%{{.*}}: i32{{.*}}, %{{.*}}: i32{{.*}}, %{{.*}}: f64{{.*}})
// CHECK:           %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           %[[VAL_10:.*]] = quake.alloca !quake.veq<?>[%{{.*}} : i64]
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           call @__nvqpp__mlirgen__statePrep_A{{.*}}(%[[VAL_10]], %[[VAL_16]]) : (!quake.veq<?>, f64) -> ()
// CHECK:           cc.scope {
// CHECK:             cc.loop while {
// CHECK:               cc.condition %{{.*}}
// CHECK:             } do {
// CHECK:                 quake.z %{{.*}}
// CHECK:                 %[[VAL_23:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:                 quake.apply<adj> @__nvqpp__mlirgen__statePrep_A{{.*}} %[[VAL_10]], %[[VAL_23]] : (!quake.veq<?>, f64) -> ()
// CHECK:                 func.call @__nvqpp__mlirgen__statePrep_A{{.*}}(%[[VAL_10]], %{{.*}}) : (!quake.veq<?>, f64) -> ()
// CHECK:               cc.continue
// CHECK:             } step {


struct run_circuit {

  auto operator()(const int n_qubits, const int n_itrs,
                  const double bmax) __qpu__ {

    cudaq::qvector q(n_qubits + 1); // last is ancilla
    auto &last_qubit = q.back();

    // State preparation
    statePrep_A{}(q, bmax);

    // Amplification Q^m_k as per evaluation schedule {m_0,m_1,..,m_k,..}
    for (int i = 0; i < n_itrs; ++i) {

      z(last_qubit);
      cudaq::adjoint(statePrep_A{}, q, bmax);
      QernelZero{}(q);
      statePrep_A{}(q, bmax);
    }
    // Measure the last ancilla qubit
    mz(last_qubit);
  }
};

// ADJOINT-LABEL:   func.func private @__nvqpp__mlirgen__statePrep_A.adj(
// ADJOINT-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: f64) {
// ADJOINT-DAG:     %[[VAL_2:.*]] = arith.constant -1 : i64
// ADJOINT-DAG:     %[[VAL_3:.*]] = arith.constant -1 : i32
// ADJOINT-DAG:     %[[VAL_4:.*]] = arith.constant 0 : i32
// ADJOINT-DAG:     %[[VAL_5:.*]] = arith.constant 2.000000e+00 : f64
// ADJOINT-DAG:     %[[VAL_6:.*]] = arith.constant 1 : i64
// ADJOINT-DAG:     %[[VAL_7:.*]] = arith.constant 0 : i64
// ADJOINT-DAG:     %[[VAL_8:.*]] = arith.constant 1 : i32
// ADJOINT:         %[[VAL_9:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// ADJOINT:         %[[VAL_10:.*]] = cc.cast %[[VAL_9]] : (i64) -> i32
// ADJOINT:         %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : i32
// ADJOINT:         %[[VAL_12:.*]] = cc.cast signed %[[VAL_11]] : (i32) -> i64
// ADJOINT:         %[[VAL_13:.*]] = arith.subi %[[VAL_12]], %[[VAL_6]] : i64
// ADJOINT:         %[[VAL_14:.*]] = quake.subveq %[[VAL_0]], 0, %[[VAL_13]] : (!quake.veq<?>, i64) -> !quake.veq<?>
// ADJOINT:         %[[VAL_15:.*]] = quake.veq_size %[[VAL_14]] : (!quake.veq<?>) -> i64
// ADJOINT:         %[[VAL_16:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : i32
// ADJOINT:         %[[VAL_17:.*]] = math.fpowi %[[VAL_5]], %[[VAL_16]] : f64, i32
// ADJOINT:         %[[VAL_18:.*]] = arith.divf %[[VAL_1]], %[[VAL_17]] : f64
// ADJOINT:         %[[VAL_19:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : i32
// ADJOINT:         %[[VAL_20:.*]] = cc.cast signed %[[VAL_19]] : (i32) -> i64
// ADJOINT:         %[[VAL_21:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:         %[[VAL_22:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : i32
// ADJOINT:         %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : i32
// ADJOINT:         %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_8]] : i32
// ADJOINT:         %[[VAL_25:.*]] = arith.cmpi sgt, %[[VAL_24]], %[[VAL_4]] : i32
// ADJOINT:         %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_24]], %[[VAL_4]] : i32
// ADJOINT:         %[[VAL_27:.*]]:2 = cc.loop while ((%[[VAL_28:.*]] = %[[VAL_26]], %[[VAL_29:.*]] = %[[VAL_26]]) -> (i32, i32)) {
// ADJOINT:           %[[VAL_30:.*]] = arith.cmpi sgt, %[[VAL_29]], %[[VAL_4]] : i32
// ADJOINT:           cc.condition %[[VAL_30]](%[[VAL_28]], %[[VAL_29]] : i32, i32)
// ADJOINT:         } do {
// ADJOINT:         ^bb0(%[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: i32):
// ADJOINT:           %[[VAL_33:.*]] = arith.subi %[[VAL_10]], %[[VAL_31]] : i32
// ADJOINT:           %[[VAL_34:.*]] = arith.subi %[[VAL_33]], %[[VAL_8]] : i32
// ADJOINT:           %[[VAL_35:.*]] = math.fpowi %[[VAL_5]], %[[VAL_34]] : f64, i32
// ADJOINT:           %[[VAL_36:.*]] = arith.divf %[[VAL_1]], %[[VAL_35]] : f64
// ADJOINT:           %[[VAL_37:.*]] = arith.subi %[[VAL_31]], %[[VAL_8]] : i32
// ADJOINT:           %[[VAL_38:.*]] = cc.cast signed %[[VAL_37]] : (i32) -> i64
// ADJOINT:           %[[VAL_39:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_38]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:           %[[VAL_40:.*]] = arith.subi %[[VAL_10]], %[[VAL_8]] : i32
// ADJOINT:           %[[VAL_41:.*]] = cc.cast signed %[[VAL_40]] : (i32) -> i64
// ADJOINT:           %[[VAL_42:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_41]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:           %[[VAL_43:.*]] = arith.negf %[[VAL_36]] : f64
// ADJOINT:           quake.ry (%[[VAL_43]]) {{\[}}%[[VAL_39]]] %[[VAL_42]] : (f64, !quake.ref, !quake.ref) -> ()
// ADJOINT:           cc.continue %[[VAL_31]], %[[VAL_32]] : i32, i32
// ADJOINT:         } step {
// ADJOINT:         ^bb0(%[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: i32):
// ADJOINT:           %[[VAL_46:.*]] = arith.subi %[[VAL_44]], %[[VAL_8]] : i32
// ADJOINT:           %[[VAL_47:.*]] = arith.subi %[[VAL_45]], %[[VAL_8]] : i32
// ADJOINT:           cc.continue %[[VAL_46]], %[[VAL_47]] : i32, i32
// ADJOINT:         }
// ADJOINT:         %[[VAL_48:.*]] = arith.negf %[[VAL_18]] : f64
// ADJOINT:         quake.ry (%[[VAL_48]]) %[[VAL_21]] : (f64, !quake.ref) -> ()
// ADJOINT:         %[[VAL_49:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : i64
// ADJOINT:         %[[VAL_50:.*]] = arith.addi %[[VAL_49]], %[[VAL_6]] : i64
// ADJOINT:         %[[VAL_51:.*]] = arith.cmpi sgt, %[[VAL_50]], %[[VAL_7]] : i64
// ADJOINT:         %[[VAL_52:.*]] = arith.select %[[VAL_51]], %[[VAL_50]], %[[VAL_7]] : i64
// ADJOINT:         %[[VAL_53:.*]] = arith.subi %[[VAL_52]], %[[VAL_6]] : i64
// ADJOINT:         %[[VAL_54:.*]]:2 = cc.loop while ((%[[VAL_55:.*]] = %[[VAL_53]], %[[VAL_56:.*]] = %[[VAL_52]]) -> (i64, i64)) {
// ADJOINT:           %[[VAL_57:.*]] = arith.cmpi sgt, %[[VAL_56]], %[[VAL_7]] : i64
// ADJOINT:           cc.condition %[[VAL_57]](%[[VAL_55]], %[[VAL_56]] : i64, i64)
// ADJOINT:         } do {
// ADJOINT:         ^bb0(%[[VAL_58:.*]]: i64, %[[VAL_59:.*]]: i64):
// ADJOINT:           %[[VAL_60:.*]] = quake.extract_ref %[[VAL_14]]{{\[}}%[[VAL_58]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:           quake.h %[[VAL_60]] : (!quake.ref) -> ()
// ADJOINT:           cc.continue %[[VAL_58]], %[[VAL_59]] : i64, i64
// ADJOINT:         } step {
// ADJOINT:         ^bb0(%[[VAL_61:.*]]: i64, %[[VAL_62:.*]]: i64):
// ADJOINT:           %[[VAL_63:.*]] = arith.subi %[[VAL_61]], %[[VAL_6]] : i64
// ADJOINT:           %[[VAL_64:.*]] = arith.subi %[[VAL_62]], %[[VAL_6]] : i64
// ADJOINT:           cc.continue %[[VAL_63]], %[[VAL_64]] : i64, i64
// ADJOINT:         }
// ADJOINT:         return
// ADJOINT:       }

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__QernelZero(

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__run_circuit(
// ADJOINT:           ^bb0(
// ADJOINT:             quake.z %
// ADJOINT:             func.call @__nvqpp__mlirgen__statePrep_A.adj(%
// ADJOINT:             func.call @__nvqpp__mlirgen__QernelZero(%
// ADJOINT:             func.call @__nvqpp__mlirgen__statePrep_A(%
