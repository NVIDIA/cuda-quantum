/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --apply-op-specialization | FileCheck %s

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

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__kernel_gamma
// CHECK-SAME:        .adj(%[[VAL_0:.*]]: !quake.ref) {
// CHECK:           cc.scope {
// CHECK:             %[[VAL_1:.*]] = arith.constant 6 : i32
// CHECK:             %[[VAL_2:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_7:.*]] = arith.subi %[[VAL_6]], %[[VAL_5]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.divsi %[[VAL_9]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_10]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_13:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_14:.*]] = arith.subi %[[VAL_12]], %[[VAL_13]] : i32
// CHECK:             %[[VAL_15:.*]] = arith.muli %[[VAL_14]], %[[VAL_7]] : i32
// CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_3]], %[[VAL_15]] : i32
// CHECK:             cc.store %[[VAL_16]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_17:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %[[VAL_12]]) -> (i32)) {
// CHECK:               %[[VAL_20:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_21:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_22:.*]] = arith.cmpi sge, %[[VAL_20]], %[[VAL_21]] : i32
// CHECK:               %[[VAL_23:.*]] = arith.cmpi sgt, %[[VAL_19]], %[[VAL_17]] : i32
// CHECK:               cc.condition %[[VAL_23]](%[[VAL_19]] : i32)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_24:.*]]: i32):
// CHECK:               cc.scope {
// CHECK:                 quake.y %[[VAL_0]]
// CHECK:                 quake.x %[[VAL_0]]
// CHECK:                 quake.h %[[VAL_0]]
// CHECK:               }
// CHECK:               cc.continue %[[VAL_24]] : i32
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_25:.*]]: i32):
// CHECK:               %[[VAL_26:.*]] = arith.constant 2 : i32
// CHECK:               %[[VAL_27:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_28:.*]] = arith.addi %[[VAL_27]], %[[VAL_26]] : i32
// CHECK:               cc.store %[[VAL_28]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_29:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_30:.*]] = arith.subi %[[VAL_25]], %[[VAL_29]] : i32
// CHECK:               cc.continue %[[VAL_30]] : i32
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func private @__nvqpp__mlirgen__kernel_alpha
// CHECK-SAME:        .adj(%[[VAL_0:.*]]: !quake.ref) {
// CHECK:           cc.scope {
// CHECK:             %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_2:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_4:.*]] = arith.constant 4 : i32
// CHECK:             %[[VAL_5:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_7:.*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : i32
// CHECK:             %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_9:.*]] = arith.select %[[VAL_8]], %[[VAL_7]], %[[VAL_6]] : i32
// CHECK:             %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_3]], %[[VAL_11]] : i32
// CHECK:             cc.store %[[VAL_12]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:             %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_14:.*]] = cc.loop while ((%[[VAL_15:.*]] = %[[VAL_9]]) -> (i32)) {
// CHECK:               %[[VAL_16:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_17:.*]] = arith.constant 4 : i32
// CHECK:               %[[VAL_18:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_17]] : i32
// CHECK:               %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_13]] : i32
// CHECK:               cc.condition %[[VAL_19]](%[[VAL_15]] : i32)
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_20:.*]]: i32):
// CHECK:               cc.scope {
// CHECK:                 quake.z %[[VAL_0]]
// CHECK:                 quake.y %[[VAL_0]]
// CHECK:                 quake.x %[[VAL_0]]
// CHECK:               }
// CHECK:               cc.continue %[[VAL_20]] : i32
// CHECK:             } step {
// CHECK:             ^bb0(%[[VAL_21:.*]]: i32):
// CHECK:               %[[VAL_22:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_23:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_24:.*]] = arith.subi %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:               cc.store %[[VAL_24]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:               %[[VAL_25:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_26:.*]] = arith.subi %[[VAL_21]], %[[VAL_25]] : i32
// CHECK:               cc.continue %[[VAL_26]] : i32
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_alpha
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_beta
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_gamma
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_delta

