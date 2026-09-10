/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// A comma expression evaluates its lhs for side-effects only and takes its
// value from the rhs.
struct multi_counter_step {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    for (int i = 0, j = 3; i < 4; i++, j--)
      h(q[0]);
    int a = 0, b = 0;
    a = (b = 2, b + 3);
    if (a == 5)
      x(q[1]);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__multi_counter_step() attributes
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 5 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_4]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:             %[[VAL_8:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_3]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_9:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_2]] : i32
// CHECK:               cc.condition %[[VAL_10]]
// CHECK:             } do {
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:               quake.h %[[VAL_11]] : (!quake.ref) -> ()
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_12:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_13]], %[[VAL_7]] : !cc.ptr<i32>
// CHECK:               %[[VAL_14:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:               %[[VAL_15:.*]] = arith.subi %[[VAL_14]], %[[VAL_1]] : i32
// CHECK:               cc.store %[[VAL_15]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_16:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_16]] : !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_17]] : !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_17]] : !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = cc.load %[[VAL_17]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i32
// CHECK:           cc.store %[[VAL_19]], %[[VAL_16]] : !cc.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i32>
// CHECK:           %[[VAL_21:.*]] = arith.cmpi eq, %[[VAL_20]], %[[VAL_0]] : i32
// CHECK:           cc.if(%[[VAL_21]]) {
// CHECK:             %[[VAL_22:.*]] = quake.extract_ref %[[VAL_6]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x %[[VAL_22]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           return
// CHECK:         }
