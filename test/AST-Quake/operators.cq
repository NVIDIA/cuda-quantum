/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

void foo(int, int);

struct integer_test {
  void operator()(int sh, unsigned us) __qpu__ {
    // Test the lowering of various C++ integral operations.
    int r1 = sh >> 1;
    int r2 = us >> 1;
    int r3 = sh << 2;
    int r4 = us << 3;
    int r5 = r1 & r2 | r3 ^ r4;
    int r6 = r1 + r2 * r3 - r4 / r5;
    foo(r5, r6);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__integer_test(
// CHECK-SAME:       %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) attributes
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = arith.shrsi %[[VAL_7]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_9:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = arith.shrui %[[VAL_10]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_12:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_11]], %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = arith.shli %[[VAL_13]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_15:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = arith.shli %[[VAL_16]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_18:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_21:.*]] = arith.andi %[[VAL_19]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_22:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = arith.xori %[[VAL_22]], %[[VAL_23]] : i32
// CHECK:           %[[VAL_25:.*]] = arith.ori %[[VAL_21]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_26:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_25]], %[[VAL_26]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = cc.load %[[VAL_9]] : !cc.ptr<i32>
// CHECK:           %[[VAL_28:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_29:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_30:.*]] = arith.muli %[[VAL_28]], %[[VAL_29]] : i32
// CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_27]], %[[VAL_30]] : i32
// CHECK:           %[[VAL_32:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_33:.*]] = cc.load %[[VAL_26]] : !cc.ptr<i32>
// CHECK:           %[[VAL_34:.*]] = arith.divsi %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:           %[[VAL_35:.*]] = arith.subi %[[VAL_31]], %[[VAL_34]] : i32
// CHECK:           %[[VAL_36:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_35]], %[[VAL_36]] : !cc.ptr<i32>
// CHECK:           %[[VAL_37:.*]] = cc.load %[[VAL_26]] : !cc.ptr<i32>
// CHECK:           %[[VAL_38:.*]] = cc.load %[[VAL_36]] : !cc.ptr<i32>
// CHECK:           call @_Z3fooii(%[[VAL_37]], %[[VAL_38]]) : (i32, i32) -> ()
// CHECK:           return

struct short_test {
  void operator()(short sh, unsigned short us) __qpu__ {
    // Test the lowering of various C++ integral operations.
    int r1 = sh >> 1;
    int r2 = us >> 1;
    int r3 = sh << 2;
    int r4 = us << 3;
    int r5 = r1 & r2 | r3 ^ r4;
    int r6 = r1 + r2 * r3 - r4 / r5;
    foo(r5, r6);

    // Integer conversion.
    unsigned r7 = r6;
    unsigned r8 = r7 >> 1;
    int r9 = r6 >> 1;
    foo(r8, r9);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__short_test(
// CHECK-SAME:        %[[VAL_0:.*]]: i16, %[[VAL_1:.*]]: i16) attributes
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 3 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = cc.alloca i16
// CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i16>
// CHECK:           %[[VAL_6:.*]] = cc.alloca i16
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<i16>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i16>
// CHECK:           %[[VAL_8:.*]] = arith.extsi %[[VAL_7]] : i16 to i32
// CHECK:           %[[VAL_9:.*]] = arith.shrsi %[[VAL_8]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_10:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i16>
// CHECK:           %[[VAL_12:.*]] = arith.extui %[[VAL_11]] : i16 to i32
// CHECK:           %[[VAL_13:.*]] = arith.shrsi %[[VAL_12]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_14:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_13]], %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i16>
// CHECK:           %[[VAL_16:.*]] = arith.extsi %[[VAL_15]] : i16 to i32
// CHECK:           %[[VAL_17:.*]] = arith.shli %[[VAL_16]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_18:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i16>
// CHECK:           %[[VAL_20:.*]] = arith.extui %[[VAL_19]] : i16 to i32
// CHECK:           %[[VAL_21:.*]] = arith.shli %[[VAL_20]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_22:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_21]], %[[VAL_22]] : !cc.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i32
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = cc.load %[[VAL_22]] : !cc.ptr<i32>
// CHECK:           %[[VAL_28:.*]] = arith.xori %[[VAL_26]], %[[VAL_27]] : i32
// CHECK:           %[[VAL_29:.*]] = arith.ori %[[VAL_25]], %[[VAL_28]] : i32
// CHECK:           %[[VAL_30:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_29]], %[[VAL_30]] : !cc.ptr<i32>
// CHECK:           %[[VAL_31:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i32>
// CHECK:           %[[VAL_32:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           %[[VAL_33:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_34:.*]] = arith.muli %[[VAL_32]], %[[VAL_33]] : i32
// CHECK:           %[[VAL_35:.*]] = arith.addi %[[VAL_31]], %[[VAL_34]] : i32
// CHECK:           %[[VAL_36:.*]] = cc.load %[[VAL_22]] : !cc.ptr<i32>
// CHECK:           %[[VAL_37:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i32>
// CHECK:           %[[VAL_38:.*]] = arith.divsi %[[VAL_36]], %[[VAL_37]] : i32
// CHECK:           %[[VAL_39:.*]] = arith.subi %[[VAL_35]], %[[VAL_38]] : i32
// CHECK:           %[[VAL_40:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_39]], %[[VAL_40]] : !cc.ptr<i32>
// CHECK:           %[[VAL_41:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i32>
// CHECK:           %[[VAL_42:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i32>
// CHECK:           call @_Z3fooii(%[[VAL_41]], %[[VAL_42]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_43:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i32>
// CHECK:           %[[VAL_44:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_43]], %[[VAL_44]] : !cc.ptr<i32>
// CHECK:           %[[VAL_45:.*]] = cc.load %[[VAL_44]] : !cc.ptr<i32>
// CHECK:           %[[VAL_46:.*]] = arith.shrui %[[VAL_45]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_47:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_46]], %[[VAL_47]] : !cc.ptr<i32>
// CHECK:           %[[VAL_48:.*]] = cc.load %[[VAL_40]] : !cc.ptr<i32>
// CHECK:           %[[VAL_49:.*]] = arith.shrsi %[[VAL_48]], %[[VAL_4]] : i32
// CHECK:           %[[VAL_50:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_49]], %[[VAL_50]] : !cc.ptr<i32>
// CHECK:           %[[VAL_51:.*]] = cc.load %[[VAL_47]] : !cc.ptr<i32>
// CHECK:           %[[VAL_52:.*]] = cc.load %[[VAL_50]] : !cc.ptr<i32>
// CHECK:           call @_Z3fooii(%[[VAL_51]], %[[VAL_52]]) : (i32, i32) -> ()
// CHECK:           return

void bar(double);

struct fp_test {
  void operator()(float f1, float f2) __qpu__ {
    // Test the lowering of various C++ fp operations.
    double d1 = f1 + f2;
    double d2 = f1 * f2;
    double d3 = f1 - d2;
    double d4 = d3 / d2;
    bar(d4);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__fp_test(
// CHECK-SAME:        %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: f32) attributes
// CHECK:           %[[VAL_2:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<f32>
// CHECK:           %[[VAL_3:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<f32>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f32>
// CHECK:           %[[VAL_6:.*]] = arith.addf %[[VAL_4]], %[[VAL_5]] : f32
// CHECK:           %[[VAL_7:.*]] = arith.extf %[[VAL_6]] : f32 to f64
// CHECK:           %[[VAL_8:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f32>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f32>
// CHECK:           %[[VAL_11:.*]] = arith.mulf %[[VAL_9]], %[[VAL_10]] : f32
// CHECK:           %[[VAL_12:.*]] = arith.extf %[[VAL_11]] : f32 to f64
// CHECK:           %[[VAL_13:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_12]], %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f32>
// CHECK:           %[[VAL_15:.*]] = arith.extf %[[VAL_14]] : f32 to f64
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_17:.*]] = arith.subf %[[VAL_15]], %[[VAL_16]] : f64
// CHECK:           %[[VAL_18:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<f64>
// CHECK:           %[[VAL_19:.*]] = cc.load %[[VAL_18]] : !cc.ptr<f64>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_13]] : !cc.ptr<f64>
// CHECK:           %[[VAL_21:.*]] = arith.divf %[[VAL_19]], %[[VAL_20]] : f64
// CHECK:           %[[VAL_22:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_21]], %[[VAL_22]] : !cc.ptr<f64>
// CHECK:           %[[VAL_23:.*]] = cc.load %[[VAL_22]] : !cc.ptr<f64>
// CHECK:           call @_Z3bard(%[[VAL_23]]) : (f64) -> ()
// CHECK:           return
