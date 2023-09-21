/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --memtoreg=quantum=0 --canonicalize --apply-op-specialization --canonicalize | FileCheck %s

#include <cudaq.h>

struct kernel_alpha {
  void operator()(cudaq::qubit &qb) __qpu__ {
    for (int i = 0; i < 4; ++i) {
      x(qb);
      y(qb);
      z(qb);
    }
  }
};

struct kernel_beta {
  void operator()() __qpu__ {
    cudaq::qubit qb;

    cudaq::adjoint(kernel_alpha{}, qb);
  }
};

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__kernel_gamma.adj(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !quake.ref) {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_5:.*]]:2 = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_1]], %[[VAL_7:.*]] = %[[VAL_2]]) -> (i32, i32)) {
// CHECK:             %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_1]] : i32
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]], %[[VAL_7]] : i32, i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             quake.y %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_9]], %[[VAL_10]] : i32, i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_11]], %[[VAL_4]] : i32
// CHECK:             %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_3]] : i32
// CHECK:             cc.continue %[[VAL_13]], %[[VAL_14]] : i32, i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

struct kernel_gamma {
  void operator()(cudaq::qubit &qb) __qpu__ {
    for (int i = 6; i >= 0; i -= 2) {
      h(qb);
      x(qb);
      y(qb);
    }
  }
};

struct kernel_delta {
  void operator()() __qpu__ {
    cudaq::qubit qb;

    cudaq::adjoint(kernel_gamma{}, qb);
  }
};

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__kernel_alpha.adj(
// CHECK-SAME:                                      %[[VAL_0:.*]]: !quake.ref) {
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3 : i32
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 4 : i32
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]]:2 = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_1]], %[[VAL_7:.*]] = %[[VAL_3]]) -> (i32, i32)) {
// CHECK:             %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_4]] : i32
// CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]], %[[VAL_7]] : i32, i32)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32):
// CHECK:             quake.z %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             quake.y %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_9]], %[[VAL_10]] : i32, i32
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32):
// CHECK:             %[[VAL_13:.*]] = arith.subi %[[VAL_11]], %[[VAL_2]] : i32
// CHECK:             %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_2]] : i32
// CHECK:             cc.continue %[[VAL_13]], %[[VAL_14]] : i32, i32
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_alpha
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_beta
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_gamma
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_delta

