/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --apply-op-specialization | FileCheck %s
// RUN: cudaq-quake %s | cudaq-opt --lambda-lifting=constant-prop=1 \
// RUN:   --canonicalize --apply-op-specialization | \
// RUN:   FileCheck --check-prefix=LAMBDA1 %s
// RUN: cudaq-quake %s | cudaq-opt --lambda-lifting=constant-prop=1 \
// RUN:   --canonicalize --apply-op-specialization | \
// RUN:   FileCheck --check-prefix=LAMBDA2 %s

#include <cudaq.h>

__qpu__ void magic_func(cudaq::qvector<> &q) {
  auto nQubits = q.size();
  for (int step = 0; step < 100; ++step) {
    for (int j = 0; j < nQubits; j++)
      rx(-.01, q[j]);
    for (int i = 0; i < nQubits - 1; i++) {
      cudaq::compute_action([&]() { x<cudaq::ctrl>(q[i], q[i + 1]); },
                            [&]() { rz(-.01, q[i + 1]); });
    }
  }
}

struct ctrlHeisenberg {
  void operator()(int nQubits) __qpu__ {
    cudaq::qubit ctrl1;
    cudaq::qvector q(nQubits);
    cudaq::control(magic_func, ctrl1, q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// CHECK-SAME:      .ctrl(%[[ARG0:.*]]: !quake.veq<?>, %[[ARG1:.*]]: !quake.veq<?>) {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 100 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG1]] : (!quake.veq<?>) -> i64
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VEQ_SIZE_0]], %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_1:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[CONSTANT_3]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[ALLOCA_2:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 cc.loop while {
// CHECK:                   %[[LOAD_1:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_0:.*]] = cc.cast signed %[[LOAD_1]] : (i32) -> i64
// CHECK:                   %[[LOAD_2:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:                   %[[CMPI_1:.*]] = arith.cmpi ult, %[[CAST_0]], %[[LOAD_2]] : i64
// CHECK:                   cc.condition %[[CMPI_1]]
// CHECK:                 } do {
// CHECK:                   %[[LOAD_3:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_1:.*]] = cc.cast signed %[[LOAD_3]] : (i32) -> i64
// CHECK:                   %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                   quake.rx (%[[CONSTANT_1]]) {{\[}}%[[ARG0]]] %[[EXTRACT_REF_0]] : (f64, !quake.veq<?>, !quake.ref) -> ()
// CHECK:                   cc.continue
// CHECK:                 } step {
// CHECK:                   %[[LOAD_4:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONSTANT_2]] : i32
// CHECK:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.scope {
// CHECK:                 %[[ALLOCA_3:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                 cc.loop while {
// CHECK:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_2:.*]] = cc.cast signed %[[LOAD_5]] : (i32) -> i64
// CHECK:                   %[[LOAD_6:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:                   %[[SUBI_0:.*]] = arith.subi %[[LOAD_6]], %[[CONSTANT_0]] : i64
// CHECK:                   %[[CMPI_2:.*]] = arith.cmpi ult, %[[CAST_2]], %[[SUBI_0]] : i64
// CHECK:                   cc.condition %[[CMPI_2]]
// CHECK:                 } do {
// CHECK:                   %[[CREATE_LAMBDA_0:.*]] = cc.create_lambda {
// CHECK:                     %[[LOAD_7:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[CAST_3:.*]] = cc.cast signed %[[LOAD_7]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_3]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     %[[LOAD_8:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[ADDI_1:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_2]] : i32
// CHECK:                     %[[CAST_4:.*]] = cc.cast signed %[[ADDI_1]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_4]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     quake.x {{\[}}%[[EXTRACT_REF_1]]] %[[EXTRACT_REF_2]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                     cc.return
// CHECK:                   } : !cc.callable<() -> ()>
// CHECK:                   %[[CREATE_LAMBDA_1:.*]] = cc.create_lambda {
// CHECK:                     %[[LOAD_9:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[ADDI_2:.*]] = arith.addi %[[LOAD_9]], %[[CONSTANT_2]] : i32
// CHECK:                     %[[CAST_5:.*]] = cc.cast signed %[[ADDI_2]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_5]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     quake.rz (%[[CONSTANT_1]]) {{\[}}%[[ARG0]]] %[[EXTRACT_REF_3]] : (f64, !quake.veq<?>, !quake.ref) -> ()
// CHECK:                     cc.return
// CHECK:                   } : !cc.callable<() -> ()>
// CHECK:                   quake.compute_action %[[CREATE_LAMBDA_0]], %[[CREATE_LAMBDA_1]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
// CHECK:                   cc.continue
// CHECK:                 } step {
// CHECK:                   %[[LOAD_10:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_3:.*]] = arith.addi %[[LOAD_10]], %[[CONSTANT_2]] : i32
// CHECK:                   cc.store %[[ADDI_3]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_11:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_4:.*]] = arith.addi %[[LOAD_11]], %[[CONSTANT_2]] : i32
// CHECK:               cc.store %[[ADDI_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_magic_func.
// CHECK-SAME:      (%[[ARG0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 100 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VEQ_SIZE_0]], %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:           cc.scope {
// CHECK:             %[[ALLOCA_1:.*]] = cc.alloca i32
// CHECK:             cc.store %[[CONSTANT_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[CONSTANT_3]] : i32
// CHECK:               cc.condition %[[CMPI_0]]
// CHECK:             } do {
// CHECK:               cc.scope {
// CHECK:                 %[[ALLOCA_2:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 cc.loop while {
// CHECK:                   %[[LOAD_1:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_0:.*]] = cc.cast signed %[[LOAD_1]] : (i32) -> i64
// CHECK:                   %[[LOAD_2:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:                   %[[CMPI_1:.*]] = arith.cmpi ult, %[[CAST_0]], %[[LOAD_2]] : i64
// CHECK:                   cc.condition %[[CMPI_1]]
// CHECK:                 } do {
// CHECK:                   %[[LOAD_3:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_1:.*]] = cc.cast signed %[[LOAD_3]] : (i32) -> i64
// CHECK:                   %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                   quake.rx (%[[CONSTANT_1]]) %[[EXTRACT_REF_0]] : (f64, !quake.ref) -> ()
// CHECK:                   cc.continue
// CHECK:                 } step {
// CHECK:                   %[[LOAD_4:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONSTANT_2]] : i32
// CHECK:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.scope {
// CHECK:                 %[[ALLOCA_3:.*]] = cc.alloca i32
// CHECK:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                 cc.loop while {
// CHECK:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                   %[[CAST_2:.*]] = cc.cast signed %[[LOAD_5]] : (i32) -> i64
// CHECK:                   %[[LOAD_6:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// CHECK:                   %[[SUBI_0:.*]] = arith.subi %[[LOAD_6]], %[[CONSTANT_0]] : i64
// CHECK:                   %[[CMPI_2:.*]] = arith.cmpi ult, %[[CAST_2]], %[[SUBI_0]] : i64
// CHECK:                   cc.condition %[[CMPI_2]]
// CHECK:                 } do {
// CHECK:                   %[[CREATE_LAMBDA_0:.*]] = cc.create_lambda {
// CHECK:                     %[[LOAD_7:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[CAST_3:.*]] = cc.cast signed %[[LOAD_7]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[CAST_3]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     %[[LOAD_8:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[ADDI_1:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_2]] : i32
// CHECK:                     %[[CAST_4:.*]] = cc.cast signed %[[ADDI_1]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[CAST_4]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     quake.x {{\[}}%[[EXTRACT_REF_1]]] %[[EXTRACT_REF_2]] : (!quake.ref, !quake.ref) -> ()
// CHECK:                     cc.return
// CHECK:                   } : !cc.callable<() -> ()>
// CHECK:                   %[[CREATE_LAMBDA_1:.*]] = cc.create_lambda {
// CHECK:                     %[[LOAD_9:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                     %[[ADDI_2:.*]] = arith.addi %[[LOAD_9]], %[[CONSTANT_2]] : i32
// CHECK:                     %[[CAST_5:.*]] = cc.cast signed %[[ADDI_2]] : (i32) -> i64
// CHECK:                     %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[CAST_5]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                     quake.rz (%[[CONSTANT_1]]) %[[EXTRACT_REF_3]] : (f64, !quake.ref) -> ()
// CHECK:                     cc.return
// CHECK:                   } : !cc.callable<() -> ()>
// CHECK:                   quake.compute_action %[[CREATE_LAMBDA_0]], %[[CREATE_LAMBDA_1]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
// CHECK:                   cc.continue
// CHECK:                 } step {
// CHECK:                   %[[LOAD_10:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                   %[[ADDI_3:.*]] = arith.addi %[[LOAD_10]], %[[CONSTANT_2]] : i32
// CHECK:                   cc.store %[[ADDI_3]], %[[ALLOCA_3]] : !cc.ptr<i32>
// CHECK:                 }
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[LOAD_11:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:               %[[ADDI_4:.*]] = arith.addi %[[LOAD_11]], %[[CONSTANT_2]] : i32
// CHECK:               cc.store %[[ADDI_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// CHECK-SAME:      %[[ARG0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[LOAD_0:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// CHECK:           %[[ALLOCA_2:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[CAST_0]] : i64]
// CHECK:           %[[INSTANTIATE_CALLABLE_0:.*]] = cc.instantiate_callable @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl_closurer(%[[ALLOCA_1]]) : (!quake.ref) -> !cc.callable<(!quake.veq<?>) -> ()>
// CHECK:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl_closurer(%[[INSTANTIATE_CALLABLE_0]], %[[ALLOCA_2]]) : (!cc.callable<(!quake.veq<?>) -> ()>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// CHECK-SAME:      .ctrl_closurer(%[[ARG0:.*]]: !cc.callable<(!quake.veq<?>) -> ()>,
// CHECK-SAME:      %[[ARG1:.*]]: !quake.veq<?>) {
// CHECK:           %[[CALLABLE_CLOSURE_0:.*]] = cc.callable_closure %[[ARG0]] : (!cc.callable<(!quake.veq<?>) -> ()>) -> !quake.ref
// CHECK:           %[[CONCAT_0:.*]] = quake.concat %[[CALLABLE_CLOSURE_0]] : (!quake.ref) -> !quake.veq<?>
// CHECK:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[CONCAT_0]], %[[ARG1]]) : (!quake.veq<?>, !quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

//===----------------------------------------------------------------------===//

// clang-format off
// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.0.adj.ctrl(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !quake.veq<?>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG2:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ARG1]] : !cc.ptr<i32>
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG2]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           %[[LOAD_1:.*]] = cc.load %[[ARG1]] : !cc.ptr<i32>
// LAMBDA1:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[CONSTANT_0]] : i32
// LAMBDA1:           %[[CAST_1:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG2]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           quake.x {{\[}}%[[ARG0]], %[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.veq<?>, !quake.ref, !quake.ref) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.0.adj(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ARG0]] : !cc.ptr<i32>
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           %[[LOAD_1:.*]] = cc.load %[[ARG0]] : !cc.ptr<i32>
// LAMBDA1:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[CONSTANT_0]] : i32
// LAMBDA1:           %[[CAST_1:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.0.ctrl(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !quake.veq<?>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG2:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ARG1]] : !cc.ptr<i32>
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG2]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           %[[LOAD_1:.*]] = cc.load %[[ARG1]] : !cc.ptr<i32>
// LAMBDA1:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[CONSTANT_0]] : i32
// LAMBDA1:           %[[CAST_1:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG2]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           quake.x {{\[}}%[[ARG0]], %[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.veq<?>, !quake.ref, !quake.ref) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA2-LABEL:   func.func private @__nvqpp__lifted.lambda.1.ctrl(
// LAMBDA2-SAME:      %[[ARG0:.*]]: !quake.veq<?>,
// LAMBDA2-SAME:      %[[ARG1:.*]]: !cc.ptr<i32>,
// LAMBDA2-SAME:      %[[ARG2:.*]]: !quake.veq<?>) {
// LAMBDA2:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA2:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// LAMBDA2:           %[[LOAD_0:.*]] = cc.load %[[ARG1]] : !cc.ptr<i32>
// LAMBDA2:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_0]], %[[CONSTANT_0]] : i32
// LAMBDA2:           %[[CAST_0:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA2:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG2]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA2:           quake.rz (%[[CONSTANT_1]]) {{\[}}%[[ARG0]]] %[[EXTRACT_REF_0]] : (f64, !quake.veq<?>, !quake.ref) -> ()
// LAMBDA2:           return
// LAMBDA2:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// LAMBDA1-SAME:      .ctrl(%[[ARG0:.*]]: !quake.veq<?>, %[[ARG1:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// LAMBDA1:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// LAMBDA1:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[CONSTANT_3:.*]] = arith.constant 100 : i32
// LAMBDA1:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// LAMBDA1:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG1]] : (!quake.veq<?>) -> i64
// LAMBDA1:           %[[ALLOCA_0:.*]] = cc.alloca i64
// LAMBDA1:           cc.store %[[VEQ_SIZE_0]], %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:           cc.scope {
// LAMBDA1:             %[[ALLOCA_1:.*]] = cc.alloca i32
// LAMBDA1:             cc.store %[[CONSTANT_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:             cc.loop while {
// LAMBDA1:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[CONSTANT_3]] : i32
// LAMBDA1:               cc.condition %[[CMPI_0]]
// LAMBDA1:             } do {
// LAMBDA1:               cc.scope {
// LAMBDA1:                 %[[ALLOCA_2:.*]] = cc.alloca i32
// LAMBDA1:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                 cc.loop while {
// LAMBDA1:                   %[[LOAD_1:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_0:.*]] = cc.cast signed %[[LOAD_1]] : (i32) -> i64
// LAMBDA1:                   %[[LOAD_2:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:                   %[[CMPI_1:.*]] = arith.cmpi ult, %[[CAST_0]], %[[LOAD_2]] : i64
// LAMBDA1:                   cc.condition %[[CMPI_1]]
// LAMBDA1:                 } do {
// LAMBDA1:                   %[[LOAD_3:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_1:.*]] = cc.cast signed %[[LOAD_3]] : (i32) -> i64
// LAMBDA1:                   %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:                   quake.rx (%[[CONSTANT_1]]) {{\[}}%[[ARG0]]] %[[EXTRACT_REF_0]] : (f64, !quake.veq<?>, !quake.ref) -> ()
// LAMBDA1:                   cc.continue
// LAMBDA1:                 } step {
// LAMBDA1:                   %[[LOAD_4:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONSTANT_2]] : i32
// LAMBDA1:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                 }
// LAMBDA1:               }
// LAMBDA1:               cc.scope {
// LAMBDA1:                 %[[ALLOCA_3:.*]] = cc.alloca i32
// LAMBDA1:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                 cc.loop while {
// LAMBDA1:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_2:.*]] = cc.cast signed %[[LOAD_5]] : (i32) -> i64
// LAMBDA1:                   %[[LOAD_6:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:                   %[[SUBI_0:.*]] = arith.subi %[[LOAD_6]], %[[CONSTANT_0]] : i64
// LAMBDA1:                   %[[CMPI_2:.*]] = arith.cmpi ult, %[[CAST_2]], %[[SUBI_0]] : i64
// LAMBDA1:                   cc.condition %[[CMPI_2]]
// LAMBDA1:                 } do {
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.0(%[[ALLOCA_3]], %[[ARG1]]) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   %[[INSTANTIATE_CALLABLE_0:.*]] = cc.instantiate_callable @__nvqpp__lifted.lambda.1.ctrl_closurev(%[[ARG0]]) : (!quake.veq<?>) -> !cc.callable<(!cc.ptr<i32>, !quake.veq<?>) -> ()>
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.1.ctrl_closurev(%[[INSTANTIATE_CALLABLE_0]], %[[ALLOCA_3]], %[[ARG1]]) : (!cc.callable<(!cc.ptr<i32>, !quake.veq<?>) -> ()>, !cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.0.adj(%[[ALLOCA_3]], %[[ARG1]]) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   cc.continue
// LAMBDA1:                 } step {
// LAMBDA1:                   %[[LOAD_7:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                   %[[ADDI_1:.*]] = arith.addi %[[LOAD_7]], %[[CONSTANT_2]] : i32
// LAMBDA1:                   cc.store %[[ADDI_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                 }
// LAMBDA1:               }
// LAMBDA1:               cc.continue
// LAMBDA1:             } step {
// LAMBDA1:               %[[LOAD_8:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:               %[[ADDI_2:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_2]] : i32
// LAMBDA1:               cc.store %[[ADDI_2]], %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:             }
// LAMBDA1:           }
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func @__nvqpp__mlirgen__function_magic_func.
// LAMBDA1-SAME:     (%[[ARG0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel", no_this} {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
// LAMBDA1:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// LAMBDA1:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[CONSTANT_3:.*]] = arith.constant 100 : i32
// LAMBDA1:           %[[CONSTANT_4:.*]] = arith.constant 0 : i32
// LAMBDA1:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
// LAMBDA1:           %[[ALLOCA_0:.*]] = cc.alloca i64
// LAMBDA1:           cc.store %[[VEQ_SIZE_0]], %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:           cc.scope {
// LAMBDA1:             %[[ALLOCA_1:.*]] = cc.alloca i32
// LAMBDA1:             cc.store %[[CONSTANT_4]], %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:             cc.loop while {
// LAMBDA1:               %[[LOAD_0:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_0]], %[[CONSTANT_3]] : i32
// LAMBDA1:               cc.condition %[[CMPI_0]]
// LAMBDA1:             } do {
// LAMBDA1:               cc.scope {
// LAMBDA1:                 %[[ALLOCA_2:.*]] = cc.alloca i32
// LAMBDA1:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                 cc.loop while {
// LAMBDA1:                   %[[LOAD_1:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_0:.*]] = cc.cast signed %[[LOAD_1]] : (i32) -> i64
// LAMBDA1:                   %[[LOAD_2:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:                   %[[CMPI_1:.*]] = arith.cmpi ult, %[[CAST_0]], %[[LOAD_2]] : i64
// LAMBDA1:                   cc.condition %[[CMPI_1]]
// LAMBDA1:                 } do {
// LAMBDA1:                   %[[LOAD_3:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_1:.*]] = cc.cast signed %[[LOAD_3]] : (i32) -> i64
// LAMBDA1:                   %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:                   quake.rx (%[[CONSTANT_1]]) %[[EXTRACT_REF_0]] : (f64, !quake.ref) -> ()
// LAMBDA1:                   cc.continue
// LAMBDA1:                 } step {
// LAMBDA1:                   %[[LOAD_4:.*]] = cc.load %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                   %[[ADDI_0:.*]] = arith.addi %[[LOAD_4]], %[[CONSTANT_2]] : i32
// LAMBDA1:                   cc.store %[[ADDI_0]], %[[ALLOCA_2]] : !cc.ptr<i32>
// LAMBDA1:                 }
// LAMBDA1:               }
// LAMBDA1:               cc.scope {
// LAMBDA1:                 %[[ALLOCA_3:.*]] = cc.alloca i32
// LAMBDA1:                 cc.store %[[CONSTANT_4]], %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                 cc.loop while {
// LAMBDA1:                   %[[LOAD_5:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                   %[[CAST_2:.*]] = cc.cast signed %[[LOAD_5]] : (i32) -> i64
// LAMBDA1:                   %[[LOAD_6:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i64>
// LAMBDA1:                   %[[SUBI_0:.*]] = arith.subi %[[LOAD_6]], %[[CONSTANT_0]] : i64
// LAMBDA1:                   %[[CMPI_2:.*]] = arith.cmpi ult, %[[CAST_2]], %[[SUBI_0]] : i64
// LAMBDA1:                   cc.condition %[[CMPI_2]]
// LAMBDA1:                 } do {
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.0(%[[ALLOCA_3]], %[[ARG0]]) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.1(%[[ALLOCA_3]], %[[ARG0]]) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   func.call @__nvqpp__lifted.lambda.0.adj(%[[ALLOCA_3]], %[[ARG0]]) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:                   cc.continue
// LAMBDA1:                 } step {
// LAMBDA1:                   %[[LOAD_7:.*]] = cc.load %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                   %[[ADDI_1:.*]] = arith.addi %[[LOAD_7]], %[[CONSTANT_2]] : i32
// LAMBDA1:                   cc.store %[[ADDI_1]], %[[ALLOCA_3]] : !cc.ptr<i32>
// LAMBDA1:                 }
// LAMBDA1:               }
// LAMBDA1:               cc.continue
// LAMBDA1:             } step {
// LAMBDA1:               %[[LOAD_8:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:               %[[ADDI_2:.*]] = arith.addi %[[LOAD_8]], %[[CONSTANT_2]] : i32
// LAMBDA1:               cc.store %[[ADDI_2]], %[[ALLOCA_1]] : !cc.ptr<i32>
// LAMBDA1:             }
// LAMBDA1:           }
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func @__nvqpp__mlirgen__ctrlHeisenberg(
// LAMBDA1-SAME:      %[[ARG0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// LAMBDA1:           %[[ALLOCA_0:.*]] = cc.alloca i32
// LAMBDA1:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32>
// LAMBDA1:           %[[ALLOCA_1:.*]] = quake.alloca !quake.ref
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// LAMBDA1:           %[[ALLOCA_2:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[CAST_0]] : i64]
// LAMBDA1:           %[[INSTANTIATE_CALLABLE_0:.*]] = cc.instantiate_callable @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl_closurer(%[[ALLOCA_1]]) : (!quake.ref) -> !cc.callable<(!quake.veq<?>) -> ()>
// LAMBDA1:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl_closurer(%[[INSTANTIATE_CALLABLE_0]], %[[ALLOCA_2]]) : (!cc.callable<(!quake.veq<?>) -> ()>, !quake.veq<?>) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__callable.thunk.lambda.1(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.callable<() -> ()>) attributes {"cudaq-kernel"} {
// LAMBDA1:           %[[CALLABLE_CLOSURE_0:.*]]:2 = cc.callable_closure %[[ARG0]] : (!cc.callable<() -> ()>) -> (!cc.ptr<i32>, !quake.veq<?>)
// LAMBDA1:           call @__nvqpp__lifted.lambda.1(%[[CALLABLE_CLOSURE_0]]#0, %[[CALLABLE_CLOSURE_0]]#1) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.1(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[CONSTANT_1:.*]] = arith.constant -1.000000e-02 : f64
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ARG0]] : !cc.ptr<i32>
// LAMBDA1:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_0]], %[[CONSTANT_0]] : i32
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           quake.rz (%[[CONSTANT_1]]) %[[EXTRACT_REF_0]] : (f64, !quake.ref) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__callable.thunk.lambda.0(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.callable<() -> ()>) attributes {"cudaq-kernel"} {
// LAMBDA1:           %[[CALLABLE_CLOSURE_0:.*]]:2 = cc.callable_closure %[[ARG0]] : (!cc.callable<() -> ()>) -> (!cc.ptr<i32>, !quake.veq<?>)
// LAMBDA1:           call @__nvqpp__lifted.lambda.0(%[[CALLABLE_CLOSURE_0]]#0, %[[CALLABLE_CLOSURE_0]]#1) : (!cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.0(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
// LAMBDA1:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
// LAMBDA1:           %[[LOAD_0:.*]] = cc.load %[[ARG0]] : !cc.ptr<i32>
// LAMBDA1:           %[[CAST_0:.*]] = cc.cast signed %[[LOAD_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_0]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           %[[LOAD_1:.*]] = cc.load %[[ARG0]] : !cc.ptr<i32>
// LAMBDA1:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[CONSTANT_0]] : i32
// LAMBDA1:           %[[CAST_1:.*]] = cc.cast signed %[[ADDI_0]] : (i32) -> i64
// LAMBDA1:           %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG1]]{{\[}}%[[CAST_1]]] : (!quake.veq<?>, i64) -> !quake.ref
// LAMBDA1:           quake.x {{\[}}%[[EXTRACT_REF_0]]] %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__mlirgen__function_magic_func.
// LAMBDA1-SAME:      .ctrl_closurer(%[[ARG0:.*]]: !cc.callable<(!quake.veq<?>) -> ()>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CALLABLE_CLOSURE_0:.*]] = cc.callable_closure %[[ARG0]] : (!cc.callable<(!quake.veq<?>) -> ()>) -> !quake.ref
// LAMBDA1:           %[[CONCAT_0:.*]] = quake.concat %[[CALLABLE_CLOSURE_0]] : (!quake.ref) -> !quake.veq<?>
// LAMBDA1:           call @__nvqpp__mlirgen__function_magic_func.{{.*}}.ctrl(%[[CONCAT_0]], %[[ARG1]]) : (!quake.veq<?>, !quake.veq<?>) -> ()
// LAMBDA1:           return
// LAMBDA1:         }

// LAMBDA1-LABEL:   func.func private @__nvqpp__lifted.lambda.1.ctrl_closurev(
// LAMBDA1-SAME:      %[[ARG0:.*]]: !cc.callable<(!cc.ptr<i32>, !quake.veq<?>) -> ()>,
// LAMBDA1-SAME:      %[[ARG1:.*]]: !cc.ptr<i32>,
// LAMBDA1-SAME:      %[[ARG2:.*]]: !quake.veq<?>) {
// LAMBDA1:           %[[CALLABLE_CLOSURE_0:.*]] = cc.callable_closure %[[ARG0]] : (!cc.callable<(!cc.ptr<i32>, !quake.veq<?>) -> ()>) -> !quake.veq<?>
// LAMBDA1:           %[[CONCAT_0:.*]] = quake.concat %[[CALLABLE_CLOSURE_0]] : (!quake.veq<?>) -> !quake.veq<?>
// LAMBDA1:           call @__nvqpp__lifted.lambda.1.ctrl(%[[CONCAT_0]], %[[ARG1]], %[[ARG2]]) : (!quake.veq<?>, !cc.ptr<i32>, !quake.veq<?>) -> ()
// LAMBDA1:           return
// LAMBDA1:         }
// clang-format on
