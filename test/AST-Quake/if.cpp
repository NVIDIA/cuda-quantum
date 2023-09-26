/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// Test the lowering of if and if-else statements.

struct kernel {
   __qpu__ int operator() (bool flag) {
      cudaq::qreg reg(2);
      if (flag) {
	 h(reg[0], reg[1]);
      }
      return 0;
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
// CHECK-SAME:       %[[VAL_0:.*]]: i1{{.*}}) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.h {{\[}}%[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32

struct kernel_else {
   __qpu__ int operator() (bool flag) {
      cudaq::qreg reg(2);
      if (flag) {
	 h(reg[0], reg[1]);
      } else {
	 x(reg[1], reg[0]);
      }
      return 0;
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_else(
// CHECK-SAME:       %[[VAL_0:.*]]: i1{{.*}}) -> i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.h {{\[}}%[[VAL_5]]] %[[VAL_6]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           } else {
// CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:             quake.x {{\[}}%[[VAL_7]]] %[[VAL_8]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           }
// CHECK:           return %[[VAL_1]] : i32
