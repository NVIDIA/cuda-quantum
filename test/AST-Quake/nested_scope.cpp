/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake --no-simplify %s | cudaq-opt --symbol-dce | FileCheck %s

#include <cudaq.h>

extern "C" {
void foo(double);
void bar(int);
}

struct test1 {
  void operator()() __qpu__ {
    int i;
    i = 0;
    {
      double i;
      i = 3.14;
      foo(i);
    }
    bar(i);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1
// CHECK-SAME: ()
// CHECK:           %[[VAL_0:.*]] = cc.alloca i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_0]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_2:.*]] = cc.alloca f64
// CHECK:             %[[VAL_3:.*]] = arith.constant 3.14
// CHECK:             cc.store %[[VAL_3]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:             %[[FUNC:.*]] = func.constant @foo : (f64) -> ()
// CHECK:             %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:             call_indirect %[[FUNC]](%[[VAL_4]]) : (f64) -> ()
// CHECK:           }
// CHECK:           %[[FUNC:.*]] = constant @bar : (i32) -> ()
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_0]] : !cc.ptr<i32>
// CHECK:           call_indirect %[[FUNC]](%[[VAL_5]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

struct test2 {
  void operator()(int i) __qpu__ {
    i = 0;
    {
      double i;
      i = 3.14;
      foo(i);
    }
    bar(i);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2
// CHECK-SAME:         (%[[VAL_0:.*]]: i32{{.*}})
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
// CHECK:           cc.store %[[VAL_2]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_3:.*]] = cc.alloca f64
// CHECK:             %[[VAL_4:.*]] = arith.constant 3.14
// CHECK:             cc.store %[[VAL_4]], %[[VAL_3]] : !cc.ptr<f64>
// CHECK:             %[[FUNC:.*]] = func.constant @foo : (f64) -> ()
// CHECK:             %[[VAL_5:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f64>
// CHECK:             call_indirect %[[FUNC]](%[[VAL_5]]) : (f64) -> ()
// CHECK:           }
// CHECK:           %[[FUNC:.*]] = constant @bar : (i32) -> ()
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           call_indirect %[[FUNC]](%[[VAL_6]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

struct test3 {
  void operator()(double i) __qpu__ {
    i = 3.14;
    for (int i = 0; i < 20; ++i) {
      bar(i);
    }
    foo(i);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3
// CHECK-SAME:         (%[[VAL_0:.*]]: f64{{.*}})
// CHECK:           %[[VAL_1:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           %[[VAL_2:.*]] = arith.constant 3.14
// CHECK:           cc.store %[[VAL_2]], %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           cc.scope {
// CHECK:             %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:             %[[VAL_4:.*]] = cc.alloca i32
// CHECK:             cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_6:.*]] = arith.constant 20 : i32
// CHECK:   %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:               cc.condition %[[VAL_7]]
// CHECK:             } do {
// CHECK:                 %[[FUNC:.*]] = func.constant @bar : (i32) -> ()
// CHECK:                 %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:                 call_indirect %[[FUNC]](%[[VAL_8]]) : (i32) -> ()
// CHECK:             } step {
// CHECK:               %[[VAL_9:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:               %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:   %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_10]] : i32
// CHECK:               cc.store %[[VAL_11]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[FUNC:.*]] = constant @foo : (f64) -> ()
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           call_indirect %[[FUNC]](%[[VAL_12]]) : (f64) -> ()
// CHECK:           return
// CHECK:         }

