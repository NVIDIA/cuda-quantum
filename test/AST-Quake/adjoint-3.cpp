/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --apply-op-specialization | FileCheck --check-prefixes=ADJOINT %s

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
// CHECK:               cc.scope {
// CHECK:                 quake.z %{{.*}}
// CHECK:                 %[[VAL_23:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:                 quake.apply<adj> @__nvqpp__mlirgen__statePrep_A{{.*}} %[[VAL_10]], %[[VAL_23]] : (!quake.veq<?>, f64) -> ()
// CHECK:                 func.call @__nvqpp__mlirgen__statePrep_A{{.*}}(%[[VAL_10]], %{{.*}}) : (!quake.veq<?>, f64) -> ()
// CHECK:               }
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

// ADJOINT-LABEL:   func.func private @__nvqpp__mlirgen__statePrep_A
// ADJOINT-SAME:        .adj(%[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: f64) {
// ADJOINT:           %[[VAL_3:.*]] = quake.vec_size %[[VAL_0]] : (!quake.veq<?>) -> i64
// ADJOINT:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i64 to i32
// ADJOINT:           %[[VAL_6:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_7:.*]] = arith.subi %[[VAL_4]], %[[VAL_6]] : i32
// ADJOINT:           %[[VAL_8:.*]] = arith.extsi %[[VAL_7]] : i32 to i64
// ADJOINT:           %[[VAL_9:.*]] = arith.constant 0 : i64
// ADJOINT:           %[[VAL_10:.*]] = arith.constant 1 : i64
// ADJOINT:           %[[VAL_11:.*]] = arith.subi %[[VAL_8]], %[[VAL_10]] : i64
// ADJOINT:           %[[VAL_12:.*]] = quake.subvec %[[VAL_0]], %[[VAL_9]], %[[VAL_11]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
// ADJOINT:           %[[VAL_13:.*]] = quake.vec_size %[[VAL_12]] : (!quake.veq<?>) -> i64
// ADJOINT:           %[[VAL_14:.*]] = arith.index_cast %[[VAL_13]] : i64 to index
// ADJOINT:           %[[VAL_15:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_16:.*]] = arith.constant 1 : index
// ADJOINT:           %[[VAL_17:.*]] = arith.constant 2.000000e+00 : f64
// ADJOINT:           %[[VAL_18:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_19:.*]] = arith.subi %[[VAL_4]], %[[VAL_18]] : i32
// ADJOINT:           %[[VAL_20:.*]] = arith.sitofp %[[VAL_19]] : i32 to f64
// ADJOINT:           %[[VAL_21:.*]] = math.powf %[[VAL_17]], %[[VAL_20]] : f64
// ADJOINT:           %[[VAL_22:.*]] = arith.divf %[[VAL_1]], %[[VAL_21]] : f64
// ADJOINT:           %[[VAL_23:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_24:.*]] = arith.subi %[[VAL_4]], %[[VAL_23]] : i32
// ADJOINT:           %[[VAL_25:.*]] = arith.extsi %[[VAL_24]] : i32 to i64
// ADJOINT:           %[[VAL_26:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_25]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:           %[[VAL_27:.*]]:2 = cc.scope -> (i32, f64) {
// ADJOINT:             %[[VAL_28:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_29:.*]] = cc.undef i32
// ADJOINT:             %[[VAL_30:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_31:.*]] = arith.constant 0 : i32
// ADJOINT:             %[[VAL_32:.*]] = arith.subi %[[VAL_4]], %[[VAL_28]] : i32
// ADJOINT:             %[[VAL_33:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_31]] : i32
// ADJOINT:             %[[VAL_35:.*]] = arith.constant -1 : i32
// ADJOINT:             %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_33]], %[[VAL_35]] : i32
// ADJOINT:             %[[VAL_37:.*]] = arith.addi %[[VAL_32]], %[[VAL_36]] : i32
// ADJOINT:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_30]] : i32
// ADJOINT:             %[[VAL_39:.*]] = arith.cmpi sgt, %[[VAL_38]], %[[VAL_31]] : i32
// ADJOINT:             %[[VAL_40:.*]] = arith.select %[[VAL_39]], %[[VAL_38]], %[[VAL_31]] : i32
// ADJOINT:             %[[VAL_41:.*]] = arith.subi %[[VAL_40]], %[[VAL_33]] : i32
// ADJOINT:             %[[VAL_42:.*]] = arith.addi %[[VAL_28]], %[[VAL_41]] : i32
// ADJOINT:             %[[VAL_43:.*]] = arith.constant 0 : i32
// ADJOINT:             %[[VAL_44:.*]]:4 = cc.loop while ((%[[VAL_45:.*]] = %[[VAL_42]], %[[VAL_46:.*]] = %[[VAL_4]], %[[VAL_47:.*]] = %[[VAL_1]], %[[VAL_48:.*]] = %[[VAL_40]]) -> (i32, i32, f64, i32)) {
// ADJOINT:               %[[VAL_49:.*]] = arith.cmpi slt, %[[VAL_45]], %[[VAL_46]] : i32
// ADJOINT:               %[[VAL_50:.*]] = arith.cmpi sgt, %[[VAL_48]], %[[VAL_43]] : i32
// ADJOINT:               cc.condition %[[VAL_50]](%[[VAL_45]], %[[VAL_46]], %[[VAL_47]], %[[VAL_48]] : i32, i32, f64, i32)
// ADJOINT:             } do {
// ADJOINT:             ^bb0(%[[VAL_51:.*]]: i32, %[[VAL_52:.*]]: i32, %[[VAL_53:.*]]: f64, %[[VAL_54:.*]]: i32):
// ADJOINT:               cc.scope {
// ADJOINT:                 %[[VAL_55:.*]] = arith.constant 2.000000e+00 : f64
// ADJOINT:                 %[[VAL_56:.*]] = arith.subi %[[VAL_52]], %[[VAL_51]] : i32
// ADJOINT:                 %[[VAL_57:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_58:.*]] = arith.subi %[[VAL_56]], %[[VAL_57]] : i32
// ADJOINT:                 %[[VAL_59:.*]] = arith.sitofp %[[VAL_58]] : i32 to f64
// ADJOINT:                 %[[VAL_60:.*]] = math.powf %[[VAL_55]], %[[VAL_59]] : f64
// ADJOINT:                 %[[VAL_61:.*]] = arith.divf %[[VAL_53]], %[[VAL_60]] : f64
// ADJOINT:                 %[[VAL_62:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_63:.*]] = arith.subi %[[VAL_51]], %[[VAL_62]] : i32
// ADJOINT:                 %[[VAL_64:.*]] = arith.extsi %[[VAL_63]] : i32 to i64
// ADJOINT:                 %[[VAL_65:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_64]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:                 %[[VAL_66:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_67:.*]] = arith.subi %[[VAL_52]], %[[VAL_66]] : i32
// ADJOINT:                 %[[VAL_68:.*]] = arith.extsi %[[VAL_67]] : i32 to i64
// ADJOINT:                 %[[VAL_69:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_68]]] : (!quake.veq<?>, i64) -> !quake.ref
// ADJOINT:                 %[[VAL_70:.*]] = arith.negf %[[VAL_61]] : f64
// ADJOINT:                 quake.ry (%[[VAL_70]]) {{\[}}%[[VAL_65]]] %[[VAL_69]] : (f64, !quake.ref, !quake.ref) -> ()
// ADJOINT:               }
// ADJOINT:               cc.continue %[[VAL_51]], %[[VAL_52]], %[[VAL_53]], %[[VAL_54]] : i32, i32, f64, i32
// ADJOINT:             } step {
// ADJOINT:             ^bb0(%[[VAL_71:.*]]: i32, %[[VAL_72:.*]]: i32, %[[VAL_73:.*]]: f64, %[[VAL_74:.*]]: i32):
// ADJOINT:               %[[VAL_75:.*]] = arith.constant 1 : i32
// ADJOINT:               %[[VAL_76:.*]] = arith.addi %[[VAL_71]], %[[VAL_75]] : i32
// ADJOINT:               %[[VAL_77:.*]] = arith.subi %[[VAL_71]], %[[VAL_75]] : i32
// ADJOINT:               %[[VAL_78:.*]] = arith.constant 1 : i32
// ADJOINT:               %[[VAL_79:.*]] = arith.subi %[[VAL_74]], %[[VAL_78]] : i32
// ADJOINT:               cc.continue %[[VAL_77]], %[[VAL_72]], %[[VAL_73]], %[[VAL_79]] : i32, i32, f64, i32
// ADJOINT:             }
// ADJOINT:             cc.continue %[[VAL_80:.*]]#1, %[[VAL_80]]#2 : i32, f64
// ADJOINT:           }
// ADJOINT:           %[[VAL_81:.*]] = arith.negf %[[VAL_22]] : f64
// ADJOINT:           quake.ry (%[[VAL_81]]) %[[VAL_26]] : (f64, !quake.ref) -> ()
// ADJOINT:           %[[VAL_82:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_83:.*]] = arith.constant 1 : index
// ADJOINT:           %[[VAL_84:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_82]] : index
// ADJOINT:           %[[VAL_85:.*]] = arith.constant -1 : index
// ADJOINT:           %[[VAL_86:.*]] = arith.select %[[VAL_84]], %[[VAL_83]], %[[VAL_85]] : index
// ADJOINT:           %[[VAL_87:.*]] = arith.addi %[[VAL_14]], %[[VAL_86]] : index
// ADJOINT:           %[[VAL_88:.*]] = arith.addi %[[VAL_87]], %[[VAL_16]] : index
// ADJOINT:           %[[VAL_89:.*]] = arith.cmpi sgt, %[[VAL_88]], %[[VAL_82]] : index
// ADJOINT:           %[[VAL_90:.*]] = arith.select %[[VAL_89]], %[[VAL_88]], %[[VAL_82]] : index
// ADJOINT:           %[[VAL_91:.*]] = arith.subi %[[VAL_90]], %[[VAL_83]] : index
// ADJOINT:           %[[VAL_92:.*]] = arith.addi %[[VAL_15]], %[[VAL_91]] : index
// ADJOINT:           %[[VAL_93:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_94:.*]]:2 = cc.loop while ((%[[VAL_95:.*]] = %[[VAL_92]], %[[VAL_96:.*]] = %[[VAL_90]]) -> (index, index)) {
// ADJOINT:             %[[VAL_97:.*]] = arith.cmpi slt, %[[VAL_95]], %[[VAL_14]] : index
// ADJOINT:             %[[VAL_98:.*]] = arith.cmpi sgt, %[[VAL_96]], %[[VAL_93]] : index
// ADJOINT:             cc.condition %[[VAL_98]](%[[VAL_95]], %[[VAL_96]] : index, index)
// ADJOINT:           } do {
// ADJOINT:           ^bb0(%[[VAL_99:.*]]: index, %[[VAL_100:.*]]: index):
// ADJOINT:             %[[VAL_101:.*]] = quake.extract_ref %[[VAL_12]]{{\[}}%[[VAL_99]]] : (!quake.veq<?>, index) -> !quake.ref
// ADJOINT:             quake.h %[[VAL_101]] : (!quake.ref) -> ()
// ADJOINT:             cc.continue %[[VAL_99]], %[[VAL_100]] : index, index
// ADJOINT:           } step {
// ADJOINT:           ^bb0(%[[VAL_102:.*]]: index, %[[VAL_103:.*]]: index):
// ADJOINT:             %[[VAL_104:.*]] = arith.addi %[[VAL_102]], %[[VAL_16]] : index
// ADJOINT:             %[[VAL_105:.*]] = arith.subi %[[VAL_102]], %[[VAL_16]] : index
// ADJOINT:             %[[VAL_106:.*]] = arith.constant 1 : index
// ADJOINT:             %[[VAL_107:.*]] = arith.subi %[[VAL_103]], %[[VAL_106]] : index
// ADJOINT:             cc.continue %[[VAL_105]], %[[VAL_107]] : index, index
// ADJOINT:           }
// ADJOINT:           return
// ADJOINT:         }

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__run_circuit
// ADJOINT:               cc.scope {
// ADJOINT:                 quake.z %{{.*}} : (!quake.ref) -> ()
// ADJOINT:                 func.call @__nvqpp__mlirgen__statePrep_A.adj(%{{.*}}, %{{.*}}) : (!quake.veq<?>, f64) -> ()
// ADJOINT:                 func.call @__nvqpp__mlirgen__QernelZero{{.*}}(%{{[0-9]+}}) :
