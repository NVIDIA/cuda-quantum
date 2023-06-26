/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct MyKernel {
  template <typename StatePrep>
  void operator()(StatePrep &&statePrep) __qpu__ {
    cudaq::qreg q(2);
    statePrep(q);
  }
};

int main() {
  auto bell = [](cudaq::qreg<> &q) __qpu__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  MyKernel k;
  k(bell);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0
// CHECK-SAME:        (%[[VAL_0:.*]]: !quake.veq<?>{{.*}}) attributes
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i32 to i64
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_2]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = arith.extsi %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_5]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_7]] : i32 to i64
// CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:           quake.x [%[[VAL_6]]] %[[VAL_9]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__instance_MyKernel
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.lambda<(!quake.veq<?>) -> ()>{{.*}}) attributes
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.extsi %[[VAL_1]] : i32 to i64
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<?>[%[[VAL_2]] : i64]
// CHECK:           call @__nvqpp__mlirgen__Z4mainE3$_0(%[[VAL_3]]) : (!quake.veq<?>) -> ()
// CHECK:           return
// CHECK:         }

