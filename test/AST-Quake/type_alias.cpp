/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | FileCheck %s

#include <cudaq.h>
#include <cstddef>

__qpu__ std::size_t kernel1(std::size_t arg) {
   cudaq::qubit q1;
   x(q1);
   return arg;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel1._Z7kernel1m(
// CHECK-SAME:      %[[VAL_0:.*]]: i64{{.*}}) -> i64
// CHECK:           %[[VAL_1:.*]] = cc.alloca i64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i64>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i64>
// CHECK:           return %[[VAL_3]] : i64
// CHECK:         }

typedef unsigned char tiny;

__qpu__ tiny kernel2(tiny arg) {
   cudaq::qubit q1;
   x(q1);
   return arg;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel2._Z7kernel2h(
// CHECK-SAME:      %[[VAL_0:.*]]: i8{{.*}}) -> i8
// CHECK:           %[[VAL_1:.*]] = cc.alloca i8
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i8>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i8>
// CHECK:           return %[[VAL_3]] : i8
// CHECK:         }

using big = long double;

__qpu__ big kernel3(big arg) {
   cudaq::qubit q1;
   x(q1);
   return arg;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel3._Z7kernel3e(
// CHECK-SAME:      %[[VAL_0:.*]]: f[[TY:[1280]+]]{{.*}}) -> f[[TY]]
// CHECK:           %[[VAL_1:.*]] = cc.alloca f[[TY]]
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<f[[TY]]>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f[[TY]]>
// CHECK:           return %[[VAL_3]] : f[[TY]]
// CHECK:         }
