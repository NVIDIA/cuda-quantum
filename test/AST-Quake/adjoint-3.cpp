/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --apply-op-specialization | FileCheck --check-prefix=ADJOINT %s

#include <cudaq.h>

struct statePrep_A {
  void operator()(cudaq::qreg<> &q, const double bmax) __qpu__ {

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
  void operator()(cudaq::qreg<> &q) __qpu__ {

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

    cudaq::qreg q(n_qubits + 1); // last is ancilla
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
// ADJOINT-DAG:       %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
// ADJOINT-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// ADJOINT-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// ADJOINT-DAG:       %[[VAL_5:.*]] = arith.constant 1 : i64
// ADJOINT-DAG:       %[[VAL_6:.*]] = arith.constant 0 : i64
// ADJOINT-DAG:       %[[VAL_7:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_8:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// ADJOINT:           %[[VAL_9:.*]] = arith.trunci %[[VAL_8]] : i64 to i32
// ADJOINT:           %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_11:.*]] = arith.extsi %[[VAL_10]] : i32 to i64
// ADJOINT:           %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[VAL_5]] : i64
// ADJOINT:           %[[VAL_13:.*]] = quake.subveq %[[VAL_0]], %[[VAL_6]], %[[VAL_12]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// ADJOINT:           %[[VAL_14:.*]] = quake.veq_size %[[VAL_13]] : (!quake.veq<?>) -> i64
// ADJOINT:           %[[VAL_15:.*]] = arith.index_cast %[[VAL_14]] : i64 to index
// ADJOINT:           %[[VAL_16:.*]] = arith.subi %[[VAL_9]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_18:.*]] = math.fpowi %[[VAL_2]], %[[VAL_16]] : f64, i32
// ADJOINT:           %[[VAL_19:.*]] = arith.divf %[[VAL_1]], %[[VAL_18]] : f64
// ADJOINT:           %[[VAL_20:.*]] = arith.subi %[[VAL_9]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_21:.*]] = arith.extsi %[[VAL_20]] : i32 to i64
// ADJOINT:           %[[VAL_22:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_21]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:           %[[VAL_23:.*]] = arith.constant 0 : i32
// ADJOINT:           %[[VAL_24:.*]] = arith.subi %[[VAL_9]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_25:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_23]] : i32
// ADJOINT:           %[[VAL_27:.*]] = arith.constant -1 : i32
// ADJOINT:           %[[VAL_28:.*]] = arith.select %[[VAL_26]], %[[VAL_25]], %[[VAL_27]] : i32
// ADJOINT:           %[[VAL_29:.*]] = arith.addi %[[VAL_24]], %[[VAL_28]] : i32
// ADJOINT:           %[[VAL_30:.*]] = arith.addi %[[VAL_29]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_31:.*]] = arith.cmpi sgt, %[[VAL_30]], %[[VAL_23]] : i32
// ADJOINT:           %[[VAL_32:.*]] = arith.select %[[VAL_31]], %[[VAL_30]], %[[VAL_23]] : i32
// ADJOINT:           %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_25]] : i32
// ADJOINT:           %[[VAL_34:.*]] = arith.addi %[[VAL_7]], %[[VAL_33]] : i32
// ADJOINT:           %[[VAL_35:.*]] = arith.constant 0 : i32
// ADJOINT:           %[[VAL_36:.*]]:4 = cc.loop while ((%[[VAL_37:.*]] = %[[VAL_34]], %[[VAL_38:.*]] = %[[VAL_9]], %[[VAL_39:.*]] = %[[VAL_1]], %[[VAL_40:.*]] = %[[VAL_32]]) -> (i32, i32, f64, i32)) {
// ADJOINT:             %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_9]] : i32
// ADJOINT:             %[[VAL_42:.*]] = arith.cmpi sgt, %[[VAL_40]], %[[VAL_35]] : i32
// ADJOINT:             cc.condition %[[VAL_42]](%[[VAL_37]], %[[VAL_9]], %[[VAL_1]], %[[VAL_40]] : i32, i32, f64, i32)
// ADJOINT:           } do {
// ADJOINT:           ^bb0(%[[VAL_43:.*]]: i32, %[[VAL_44:.*]]: i32, %[[VAL_45:.*]]: f64, %[[VAL_46:.*]]: i32):
// ADJOINT:             %[[VAL_47:.*]] = arith.subi %[[VAL_9]], %[[VAL_43]] : i32
// ADJOINT:             %[[VAL_48:.*]] = arith.subi %[[VAL_47]], %[[VAL_7]] : i32
// ADJOINT:             %[[VAL_50:.*]] = math.fpowi %[[VAL_2]], %[[VAL_48]] : f64, i32
// ADJOINT:             %[[VAL_51:.*]] = arith.divf %[[VAL_1]], %[[VAL_50]] : f64
// ADJOINT:             %[[VAL_52:.*]] = arith.subi %[[VAL_43]], %[[VAL_7]] : i32
// ADJOINT:             %[[VAL_53:.*]] = arith.extsi %[[VAL_52]] : i32 to i64
// ADJOINT:             %[[VAL_54:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_53]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:             %[[VAL_55:.*]] = arith.subi %[[VAL_9]], %[[VAL_7]] : i32
// ADJOINT:             %[[VAL_56:.*]] = arith.extsi %[[VAL_55]] : i32 to i64
// ADJOINT:             %[[VAL_57:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_56]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:             %[[VAL_58:.*]] = arith.negf %[[VAL_51]] : f64
// ADJOINT:             quake.ry (%[[VAL_58]]) {{\[}}%[[VAL_54]]] %[[VAL_57]] : (f64, !quake.ref, !quake.ref) -> ()
// ADJOINT:             cc.continue %[[VAL_43]], %[[VAL_44]], %[[VAL_45]], %[[VAL_46]] : i32, i32, f64, i32
// ADJOINT:           } step {
// ADJOINT:           ^bb0(%[[VAL_59:.*]]: i32, %[[VAL_60:.*]]: i32, %[[VAL_61:.*]]: f64, %[[VAL_62:.*]]: i32):
// ADJOINT:             %[[VAL_63:.*]] = arith.addi %[[VAL_59]], %[[VAL_7]] : i32
// ADJOINT:             %[[VAL_64:.*]] = arith.subi %[[VAL_59]], %[[VAL_7]] : i32
// ADJOINT:             %[[VAL_65:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_66:.*]] = arith.subi %[[VAL_62]], %[[VAL_65]] : i32
// ADJOINT:             cc.continue %[[VAL_64]], %[[VAL_9]], %[[VAL_1]], %[[VAL_66]] : i32, i32, f64, i32
// ADJOINT:           }
// ADJOINT:           %[[VAL_67:.*]] = arith.negf %[[VAL_19]] : f64
// ADJOINT:           quake.ry (%[[VAL_67]]) %[[VAL_22]] : (f64, !quake.ref) -> ()
// ADJOINT:           %[[VAL_68:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_69:.*]] = arith.constant 1 : index
// ADJOINT:           %[[VAL_70:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VAL_68]] : index
// ADJOINT:           %[[VAL_71:.*]] = arith.constant -1 : index
// ADJOINT:           %[[VAL_72:.*]] = arith.select %[[VAL_70]], %[[VAL_69]], %[[VAL_71]] : index
// ADJOINT:           %[[VAL_73:.*]] = arith.addi %[[VAL_15]], %[[VAL_72]] : index
// ADJOINT:           %[[VAL_74:.*]] = arith.addi %[[VAL_73]], %[[VAL_3]] : index
// ADJOINT:           %[[VAL_75:.*]] = arith.cmpi sgt, %[[VAL_74]], %[[VAL_68]] : index
// ADJOINT:           %[[VAL_76:.*]] = arith.select %[[VAL_75]], %[[VAL_74]], %[[VAL_68]] : index
// ADJOINT:           %[[VAL_77:.*]] = arith.subi %[[VAL_76]], %[[VAL_69]] : index
// ADJOINT:           %[[VAL_78:.*]] = arith.addi %[[VAL_4]], %[[VAL_77]] : index
// ADJOINT:           %[[VAL_79:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_80:.*]]:2 = cc.loop while ((%[[VAL_81:.*]] = %[[VAL_78]], %[[VAL_82:.*]] = %[[VAL_76]]) -> (index, index)) {
// ADJOINT:             %[[VAL_83:.*]] = arith.cmpi slt, %[[VAL_81]], %[[VAL_15]] : index
// ADJOINT:             %[[VAL_84:.*]] = arith.cmpi sgt, %[[VAL_82]], %[[VAL_79]] : index
// ADJOINT:             cc.condition %[[VAL_84]](%[[VAL_81]], %[[VAL_82]] : index, index)
// ADJOINT:           } do {
// ADJOINT:           ^bb0(%[[VAL_85:.*]]: index, %[[VAL_86:.*]]: index):
// ADJOINT:             %[[VAL_87:.*]] = quake.extract_ref %[[VAL_13]]{{\[}}%[[VAL_85]]] : (!quake.veq<?>, index) -> !quake.ref
// ADJOINT:             quake.h %[[VAL_87]] : (!quake.ref) -> ()
// ADJOINT:             cc.continue %[[VAL_85]], %[[VAL_86]] : index, index
// ADJOINT:           } step {
// ADJOINT:           ^bb0(%[[VAL_88:.*]]: index, %[[VAL_89:.*]]: index):
// ADJOINT:             %[[VAL_90:.*]] = arith.addi %[[VAL_88]], %[[VAL_3]] : index
// ADJOINT:             %[[VAL_91:.*]] = arith.subi %[[VAL_88]], %[[VAL_3]] : index
// ADJOINT:             %[[VAL_92:.*]] = arith.constant 1 : index
// ADJOINT:             %[[VAL_93:.*]] = arith.subi %[[VAL_89]], %[[VAL_92]] : index
// ADJOINT:             cc.continue %[[VAL_91]], %[[VAL_93]] : index, index
// ADJOINT:           }
// ADJOINT:           return
// ADJOINT:         }

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__QernelZero(

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__run_circuit(
// ADJOINT:           ^bb0(
// ADJOINT:             quake.z %
// ADJOINT:             func.call @__nvqpp__mlirgen__statePrep_A.adj(%
// ADJOINT:             func.call @__nvqpp__mlirgen__QernelZero(%
// ADJOINT:             func.call @__nvqpp__mlirgen__statePrep_A(%
