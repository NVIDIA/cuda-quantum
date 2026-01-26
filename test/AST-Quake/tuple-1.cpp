/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <string>
#include <tuple>

void prepQubit(std::pair<int, double> basis, cudaq::qubit &q) __qpu__ {}

void RzArcTan2(bool input, std::pair<int, double> basis) __qpu__ {
  cudaq::qubit aux;
  cudaq::qubit resource;
  cudaq::qubit target;
  if (input) {
    x(target);
  }
  prepQubit(basis, target);
}

int main1() {
  RzArcTan2(true, {});
  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_prepQubit
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.struct<{i32, f64} [128,8]>, %[[VAL_1:.*]]: !quake.ref)
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<{i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<!cc.struct<{i32, f64} [128,8]>>
// CHECK:           return

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_RzArcTan2
// CHECK-SAME:        (%[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: !cc.struct<{i32, f64} [128,8]>) attributes
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<{i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<!cc.struct<{i32, f64} [128,8]>>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             quake.x %[[VAL_6]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.struct<{i32, f64} [128,8]>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_3]] : !cc.ptr<!cc.struct<{i32, f64} [128,8]>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_8]] : !cc.ptr<!cc.struct<{i32, f64} [128,8]>>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_8]] : !cc.ptr<!cc.struct<{i32, f64} [128,8]>>
// CHECK:           call @__nvqpp__mlirgen__function_prepQubit{{.*}}(%[[VAL_10]], %[[VAL_6]]) : (!cc.struct<{i32, f64} [128,8]>, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }

void prepQubit2(std::tuple<bool, float, unsigned> basis,
               cudaq::qubit &q) __qpu__ {}

void RzArcTan22(bool input, std::tuple<bool, float, unsigned> basis) __qpu__ {
  cudaq::qubit aux;
  cudaq::qubit resource;
  cudaq::qubit target;
  if (input) {
    x(target);
  }
  prepQubit2(basis, target);
}

int main2() {
  RzArcTan22(true, {});
  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_prepQubit2
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.struct<{[[TUP:.*, .*, .*]]}{{.*}}>, %[[VAL_1:.*]]: !quake.ref)
// CHECK:           %[[VAL_2:.*]] = cc.alloca !cc.struct<{[[TUP]]}{{.*}}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<!cc.struct<{[[TUP]]}{{.*}}>>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_RzArcTan22
// CHECK-SAME:        (%[[VAL_0:.*]]: i1, %[[VAL_1:.*]]: !cc.struct<{[[TUP]]}{{.*}}>) attributes
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<{[[TUP]]}{{.*}}>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<!cc.struct<{[[TUP]]}{{.*}}>>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             quake.x %[[VAL_6]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.struct<{[[TUP]]}{{.*}}>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_3]] : !cc.ptr<!cc.struct<{[[TUP]]}{{.*}}>>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_8]] : !cc.ptr<!cc.struct<{[[TUP]]}{{.*}}>>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_8]] : !cc.ptr<!cc.struct<{[[TUP]]}{{.*}}>>
// CHECK:           call @__nvqpp__mlirgen__function_prepQubit2{{.*}}(%[[VAL_10]], %[[VAL_6]]) : (!cc.struct<{[[TUP]]}{{.*}}>, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
