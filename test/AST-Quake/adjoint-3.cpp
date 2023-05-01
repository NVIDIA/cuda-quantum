/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --apply-op-specialization | FileCheck --check-prefixes=ADJOINT %s

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
// CHECK-SAME:        (%{{.*}}: i32, %{{.*}}: i32, %{{.*}}: f64)
// CHECK:           %[[VAL_5:.*]] = memref.alloca() : memref<f64>
// CHECK:           %[[VAL_10:.*]] = quake.alloca[%{{.*}} : i64] !quake.qvec<?>
// CHECK:           %[[VAL_16:.*]] = memref.load %[[VAL_5]][] : memref<f64>
// CHECK:           call @__nvqpp__mlirgen__statePrep_A{{.*}}(%[[VAL_10]], %[[VAL_16]]) : (!quake.qvec<?>, f64) -> ()
// CHECK:           cc.scope {
// CHECK:             cc.loop while {
// CHECK:               cc.condition %{{.*}}
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 quake.z %{{.*}}
// CHECK:                 %[[VAL_23:.*]] = memref.load %[[VAL_5]][] : memref<f64>
// CHECK:                 quake.apply<adj> @__nvqpp__mlirgen__statePrep_A{{.*}} %[[VAL_10]], %[[VAL_23]] : (!quake.qvec<?>, f64) -> ()
// CHECK:                 func.call @__nvqpp__mlirgen__statePrep_A{{.*}}(%[[VAL_10]], %{{.*}}) : (!quake.qvec<?>, f64) -> ()
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
// ADJOINT-SAME:        .adj(%[[VAL_0:.*]]: !quake.qvec<?>, %[[VAL_1:.*]]: f64) {
// ADJOINT:           %[[VAL_2:.*]] = memref.alloca() : memref<f64>
// ADJOINT:           memref.store %[[VAL_1]], %[[VAL_2]][] : memref<f64>
// ADJOINT:           %[[VAL_3:.*]] = quake.vec_size %[[VAL_0]] : (!quake.qvec<?>) -> i64
// ADJOINT:           %[[VAL_4:.*]] = arith.trunci %[[VAL_3]] : i64 to i32
// ADJOINT:           %[[VAL_5:.*]] = memref.alloca() : memref<i32>
// ADJOINT:           memref.store %[[VAL_4]], %[[VAL_5]][] : memref<i32>
// ADJOINT:           %[[VAL_6:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:           %[[VAL_7:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_8:.*]] = arith.subi %[[VAL_6]], %[[VAL_7]] : i32
// ADJOINT:           %[[VAL_9:.*]] = arith.extsi %[[VAL_8]] : i32 to i64
// ADJOINT:           %[[VAL_10:.*]] = arith.constant 0 : i64
// ADJOINT:           %[[VAL_11:.*]] = arith.constant 1 : i64
// ADJOINT:           %[[VAL_12:.*]] = arith.subi %[[VAL_9]], %[[VAL_11]] : i64
// ADJOINT:           %[[VAL_13:.*]] = quake.subvec %[[VAL_0]], %[[VAL_10]], %[[VAL_12]] : (!quake.qvec<?>, i64, i64) -> !quake.qvec<?>
// ADJOINT:           %[[VAL_16:.*]] = quake.vec_size %[[VAL_13]] : (!quake.qvec<?>) -> i64
// ADJOINT:           %[[VAL_17:.*]] = arith.index_cast %[[VAL_16]] : i64 to index
// ADJOINT:           %[[VAL_14:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_15:.*]] = arith.constant 1 : index
// ADJOINT:           %[[VAL_18:.*]] = memref.load %[[VAL_2]][] : memref<f64>
// ADJOINT:           %[[VAL_19:.*]] = arith.constant 2.0{{.*}} : f64
// ADJOINT:           %[[VAL_20:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:           %[[VAL_21:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_22:.*]] = arith.subi %[[VAL_20]], %[[VAL_21]] : i32
// ADJOINT:           %[[VAL_23:.*]] = arith.sitofp %[[VAL_22]] : i32 to f64
// ADJOINT:           %[[VAL_24:.*]] = math.powf %[[VAL_19]], %[[VAL_23]] : f64
// ADJOINT:           %[[VAL_25:.*]] = arith.divf %[[VAL_18]], %[[VAL_24]] : f64
// ADJOINT:           %[[VAL_26:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:           %[[VAL_27:.*]] = arith.constant 1 : i32
// ADJOINT:           %[[VAL_28:.*]] = arith.subi %[[VAL_26]], %[[VAL_27]] : i32
// ADJOINT:           %[[VAL_29:.*]] = arith.extsi %[[VAL_28]] : i32 to i64
// ADJOINT:           %[[VAL_30:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_29]]] : (!quake.qvec<?>, i64) -> !quake.qref
// ADJOINT:           cc.scope {
// ADJOINT:             %[[VAL_31:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_32:.*]] = memref.alloca() : memref<i32>
// ADJOINT:             memref.store %[[VAL_31]], %[[VAL_32]][] : memref<i32>
// ADJOINT:             %[[VAL_33:.*]] = memref.load %[[VAL_32]][] : memref<i32>
// ADJOINT:             %[[VAL_34:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:             %[[VAL_35:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_36:.*]] = arith.constant 0 : i32
// ADJOINT:             %[[VAL_37:.*]] = arith.subi %[[VAL_34]], %[[VAL_33]] : i32
// ADJOINT:             %[[VAL_38:.*]] = arith.cmpi sgt, %[[VAL_37]], %[[VAL_36]] : i32
// ADJOINT:             %[[VAL_39:.*]] = arith.select %[[VAL_38]], %[[VAL_37]], %[[VAL_36]] : i32
// ADJOINT:             %[[VAL_40:.*]] = arith.constant 1 : i32
// ADJOINT:             %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_40]] : i32
// ADJOINT:             %[[VAL_42:.*]] = arith.addi %[[VAL_33]], %[[VAL_41]] : i32
// ADJOINT:             memref.store %[[VAL_42]], %[[VAL_32]][] : memref<i32>
// ADJOINT:             %[[VAL_43:.*]] = arith.constant 0 : i32
// ADJOINT:             %[[VAL_44:.*]] = cc.loop while ((%[[VAL_45:.*]] = %[[VAL_39]]) -> (i32)) {
// ADJOINT:               %[[VAL_46:.*]] = memref.load %[[VAL_32]][] : memref<i32>
// ADJOINT:               %[[VAL_47:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:               %[[VAL_48:.*]] = arith.cmpi slt, %[[VAL_46]], %[[VAL_47]] : i32
// ADJOINT:               %[[VAL_49:.*]] = arith.cmpi sgt, %[[VAL_45]], %[[VAL_43]] : i32
// ADJOINT:               cc.condition %[[VAL_49]](%[[VAL_45]] : i32)
// ADJOINT:             } do {
// ADJOINT:             ^bb0(%[[VAL_50:.*]]: i32):
// ADJOINT:               cc.scope {
// ADJOINT:                 %[[VAL_51:.*]] = memref.load %[[VAL_2]][] : memref<f64>
// ADJOINT:                 %[[VAL_52:.*]] = arith.constant 2.0{{.*}} : f64
// ADJOINT:                 %[[VAL_53:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:                 %[[VAL_54:.*]] = memref.load %[[VAL_32]][] : memref<i32>
// ADJOINT:                 %[[VAL_55:.*]] = arith.subi %[[VAL_53]], %[[VAL_54]] : i32
// ADJOINT:                 %[[VAL_56:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_57:.*]] = arith.subi %[[VAL_55]], %[[VAL_56]] : i32
// ADJOINT:                 %[[VAL_58:.*]] = arith.sitofp %[[VAL_57]] : i32 to f64
// ADJOINT:                 %[[VAL_59:.*]] = math.powf %[[VAL_52]], %[[VAL_58]] : f64
// ADJOINT:                 %[[VAL_60:.*]] = arith.divf %[[VAL_51]], %[[VAL_59]] : f64
// ADJOINT:                 %[[VAL_61:.*]] = memref.load %[[VAL_32]][] : memref<i32>
// ADJOINT:                 %[[VAL_62:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_63:.*]] = arith.subi %[[VAL_61]], %[[VAL_62]] : i32
// ADJOINT:                 %[[VAL_64:.*]] = arith.extsi %[[VAL_63]] : i32 to i64
// ADJOINT:                 %[[VAL_65:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_64]]] : (!quake.qvec<?>, i64) -> !quake.qref
// ADJOINT:                 %[[VAL_66:.*]] = memref.load %[[VAL_5]][] : memref<i32>
// ADJOINT:                 %[[VAL_67:.*]] = arith.constant 1 : i32
// ADJOINT:                 %[[VAL_68:.*]] = arith.subi %[[VAL_66]], %[[VAL_67]] : i32
// ADJOINT:                 %[[VAL_69:.*]] = arith.extsi %[[VAL_68]] : i32 to i64
// ADJOINT:                 %[[VAL_70:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_69]]] : (!quake.qvec<?>, i64) -> !quake.qref
// ADJOINT:                 %[[VAL_71:.*]] = arith.negf %[[VAL_60]] : f64
// ADJOINT:                 quake.ry (%[[VAL_71]]) [%[[VAL_65]]] %[[VAL_70]] : (f64, !quake.qref, !quake.qref) -> ()
// ADJOINT:               }
// ADJOINT:               cc.continue %[[VAL_50]] : i32
// ADJOINT:             } step {
// ADJOINT:             ^bb0(%[[VAL_72:.*]]: i32):
// ADJOINT:               %[[VAL_73:.*]] = memref.load %[[VAL_32]][] : memref<i32>
// ADJOINT:               %[[VAL_74:.*]] = arith.constant 1 : i32
// ADJOINT:               %[[VAL_75:.*]] = arith.subi %[[VAL_73]], %[[VAL_74]] : i32
// ADJOINT:               memref.store %[[VAL_75]], %[[VAL_32]][] : memref<i32>
// ADJOINT:               %[[VAL_76:.*]] = arith.constant 1 : i32
// ADJOINT:               %[[VAL_77:.*]] = arith.subi %[[VAL_72]], %[[VAL_76]] : i32
// ADJOINT:               cc.continue %[[VAL_77]] : i32
// ADJOINT:             }
// ADJOINT:           }
// ADJOINT:           %[[VAL_78:.*]] = arith.negf %[[VAL_25]] : f64
// ADJOINT:           quake.ry (%[[VAL_78]]) %[[VAL_30]] : (f64, !quake.qref) -> ()
// ADJOINT:           %[[VAL_79:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_80:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_79]] : index
// ADJOINT:           %[[VAL_81:.*]] = arith.select %[[VAL_80]], %[[VAL_17]], %[[VAL_79]] : index
// ADJOINT:           %[[VAL_82:.*]] = arith.constant 1 : index
// ADJOINT:           %[[VAL_83:.*]] = arith.subi %[[VAL_81]], %[[VAL_82]] : index
// ADJOINT:           %[[VAL_84:.*]] = arith.addi %[[VAL_14]], %[[VAL_83]] : index
// ADJOINT:           %[[VAL_85:.*]] = arith.constant 0 : index
// ADJOINT:           %[[VAL_86:.*]]:2 = cc.loop while ((%[[VAL_87:.*]] = %[[VAL_84]], %[[VAL_88:.*]] = %[[VAL_81]]) -> (index, index)) {
// ADJOINT:             %[[VAL_89:.*]] = arith.cmpi slt, %[[VAL_87]], %[[VAL_17]] : index
// ADJOINT:             %[[VAL_90:.*]] = arith.cmpi sgt, %[[VAL_88]], %[[VAL_85]] : index
// ADJOINT:             cc.condition %[[VAL_90]](%[[VAL_87]], %[[VAL_88]] : index, index)
// ADJOINT:           } do {
// ADJOINT:           ^bb0(%[[VAL_91:.*]]: index, %[[VAL_92:.*]]: index):
// ADJOINT:             %[[VAL_93:.*]] = quake.extract_ref %[[VAL_13]][%[[VAL_91]]] : (!quake.qvec<?>, index) -> !quake.qref
// ADJOINT:             quake.h %[[VAL_93]] : (!quake.qref) -> ()
// ADJOINT:             cc.continue %[[VAL_91]], %[[VAL_92]] : index, index
// ADJOINT:           } step {
// ADJOINT:           ^bb0(%[[VAL_94:.*]]: index, %[[VAL_95:.*]]: index):
// ADJOINT:             %[[VAL_96:.*]] = arith.addi %[[VAL_94]], %[[VAL_15]] : index
// ADJOINT:             %[[VAL_97:.*]] = arith.subi %[[VAL_94]], %[[VAL_15]] : index
// ADJOINT:             %[[VAL_98:.*]] = arith.constant 1 : index
// ADJOINT:             %[[VAL_99:.*]] = arith.subi %[[VAL_95]], %[[VAL_98]] : index
// ADJOINT:             cc.continue %[[VAL_97]], %[[VAL_99]] : index, index
// ADJOINT:           }
// ADJOINT:           return
// ADJOINT:         }

// ADJOINT-LABEL:   func.func @__nvqpp__mlirgen__run_circuit
// ADJOINT:               cc.scope {
// ADJOINT:                 quake.z %{{.*}} : (!quake.qref) -> ()
// ADJOINT:                 func.call @__nvqpp__mlirgen__statePrep_A.adj(%{{.*}}, %{{.*}}) : (!quake.qvec<?>, f64) -> ()
// ADJOINT:                 func.call @__nvqpp__mlirgen__QernelZero{{.*}}(%{{[0-9]+}}) :
