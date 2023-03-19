/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

void foo(int, int);

struct integer_test {
    void operator() (short sh, unsigned short us) __qpu__ {
        // Test the lowering of various C++ integral operations.
        int r1 = sh >> 1;
        int r2 = us >> 1;
        int r3 = sh << 2;
        int r4 = us << 3;
        int r5 = r1 & r2 | r3 ^ r4;
        int r6 = r1 + r2 * r3 - r4 / r5;
        foo(r5, r6);

	// TODO: add support for integer conversion.
	//unsigned r7 = r6 >> 1;
	//int r8 = r6 >> 1;
	//foo(r7, r8);
    }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__integer_test
// CHECK:           %[[VAL_5:.*]] = arith.extsi %{{.*}} : i16 to i32
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_7:.*]] = arith.shrsi %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:           memref.store
// CHECK:           %[[VAL_10:.*]] = arith.extui %{{.*}} : i16 to i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_12:.*]] = arith.shrsi %[[VAL_10]], %[[VAL_11]] : i32
// CHECK:           memref.store
// CHECK:           %[[VAL_15:.*]] = arith.extsi %{{.*}} : i16 to i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_17:.*]] = arith.shli %[[VAL_15]], %[[VAL_16]] : i32
// CHECK:           memref.store
// CHECK:           %[[VAL_20:.*]] = arith.extui %{{.*}} : i16 to i32
// CHECK:           %[[VAL_21:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_22:.*]] = arith.shli %[[VAL_20]], %[[VAL_21]] : i32
// CHECK:           memref.store
// CHECK:           %[[VAL_26:.*]] = arith.andi %{{.*}}, %{{.*}} : i32
// CHECK:           %[[VAL_29:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK:           %[[VAL_30:.*]] = arith.ori %[[VAL_26]], %[[VAL_29]] : i32
// CHECK:           memref.store
// CHECK:           %[[VAL_35:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:           %[[VAL_36:.*]] = arith.addi %{{.*}}, %[[VAL_35]] : i32
// CHECK:           %[[VAL_39:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK:           %[[VAL_40:.*]] = arith.subi %[[VAL_36]], %[[VAL_39]] : i32
// CHECK:           memref.store
// CHECK:           call @_Z3fooii(%{{.*}}, %{{.*}}) : (i32, i32) -> ()
// CHECK:           return

void bar(double);

struct fp_test {
    void operator() (float f1, float f2) __qpu__ {
        // Test the lowering of various C++ fp operations.
        double d1 = f1 + f2;
        double d2 = f1 * f2;
        double d3 = f1 - d2;
        double d4 = d3 / d2;
        bar(d4);
    }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__fp_test
// CHECK:           %[[VAL_6:.*]] = arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:           %[[VAL_7:.*]] = arith.extf %[[VAL_6]] : f32 to f64
// CHECK:           memref.store
// CHECK:           %[[VAL_11:.*]] = arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:           %[[VAL_12:.*]] = arith.extf %[[VAL_11]] : f32 to f64
// CHECK:           memref.store
// CHECK:           %[[VAL_15:.*]] = arith.extf %{{.*}} : f32 to f64
// CHECK:           %[[VAL_17:.*]] = arith.subf %[[VAL_15]], %{{.*}} : f64
// CHECK:           memref.store
// CHECK:           %[[VAL_21:.*]] = arith.divf %{{.*}}, %{{.*}} : f64
// CHECK:           memref.store
// CHECK:           call @_Z3bard(%{{.*}}) : (f64) -> ()
// CHECK:           return
