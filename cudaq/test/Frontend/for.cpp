/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

// A `for` statement with an empty init clause (the loop variable is declared
// as a separate statement before the loop) must still emit a `step` region
// for its increment clause. This regression-tests a bug where the step
// builder was only threaded through cc.LoopOp::create in the branch that
// handles a non-empty init clause, silently dropping `++i` whenever the init
// clause was empty.
__qpu__ int foo(bool returnEarly) {
  int i = 0;

  for (; i < 4; ++i) {
    if (returnEarly)
      return 1;
  }

  return i;
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo{{.*}}(
// CHECK-SAME:      %[[VAL_0:.*]]: i1) -> i32 attributes
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           cc.loop while {
// CHECK:             %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_2]] : i32
// CHECK:             cc.condition %[[VAL_7]]
// CHECK:           } do {
// CHECK:             %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i1>
// CHECK:             cc.if(%[[VAL_8]]) {
// CHECK:               cc.unwind_return %[[VAL_1]] : i32
// CHECK:             }
// CHECK:             cc.continue
// CHECK:           } step {
// CHECK:             %[[VAL_9:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i32
// CHECK:             cc.store %[[VAL_10]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           }
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_11]] : i32
// CHECK:         }
