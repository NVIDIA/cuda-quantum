/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel
// CHECK-SAME: (%[[VAL_0:.*]]: i1) -> i32
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i1>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i1>
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca[%[[VAL_3]] : i64] !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_1]][] : memref<i1>
// CHECK:           cc.if(%[[VAL_5]]) {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_8:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_7]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_10:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_10]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.h [%[[VAL_8]]] %[[VAL_11]] :
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_12]] : i32
// CHECK:         }

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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_else
// CHECK-SAME:        (%[[VAL_0:.*]]: i1) -> i32
// CHECK:           %[[VAL_1:.*]] = memref.alloca() : memref<i1>
// CHECK:           memref.store %[[VAL_0]], %[[VAL_1]][] : memref<i1>
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_3:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_4:.*]] = quake.alloca[%[[VAL_3]] : i64] !quake.veq<?>
// CHECK:           %[[VAL_5:.*]] = memref.load %[[VAL_1]][] : memref<i1>
// CHECK:           cc.if(%[[VAL_5]]) {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_6:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_7:.*]] = arith.extsi %[[VAL_6]] : i32 to i64
// CHECK:               %[[VAL_8:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_7]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_10:.*]] = arith.extsi %[[VAL_9]] : i32 to i64
// CHECK:               %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_10]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.h [%[[VAL_8]]] %[[VAL_11]] :
// CHECK:             }
// CHECK:           } else {
// CHECK:             cc.scope {
// CHECK:               %[[VAL_12:.*]] = arith.constant 1 : i32
// CHECK:               %[[VAL_13:.*]] = arith.extsi %[[VAL_12]] : i32 to i64
// CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:               %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i32 to i64
// CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]][%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:               quake.x [%[[VAL_14]]] %[[VAL_17]] :
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : i32
// CHECK:           return %[[VAL_18]] : i32
// CHECK:         }
