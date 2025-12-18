/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct ResultThing {
  bool m1;
  bool m2_ok;
  bool m2;
  int x;
};

struct ArgumentThing {
  int i;
  int j;
  double d;
};

void debug_the_thing(int first, int second, double cola) {
  cola = cola + first + second;
}

ResultThing help_me_help_you();

struct S0 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    return help_me_help_you();
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S0(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_2:.*]] = cc.cast %[[VAL_1]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_1]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_1]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_3]], %[[VAL_5]], %[[VAL_7]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_8:.*]] = call @_Z16help_me_help_youv() : () -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           return %[[VAL_8]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// clang-format on

struct S1 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    ResultThing result = {true, false, false, 42};
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S1(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 42 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_6]], %[[VAL_8]], %[[VAL_10]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.alloca !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_12]] : !cc.ptr<i1>
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_13]] : !cc.ptr<i1>
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_11]][2] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_14]] : !cc.ptr<i1>
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_11]][3] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           return %[[VAL_16]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:         }
// clang-format on

struct S2 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    return ResultThing{true, false, true, 44};
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S2(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 44 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_6]], %[[VAL_8]], %[[VAL_10]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.alloca !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_12]] : !cc.ptr<i1>
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_13]] : !cc.ptr<i1>
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_11]][2] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_14]] : !cc.ptr<i1>
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_11]][3] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           return %[[VAL_16]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:         }
// clang-format on

struct S3 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    return {false, true, false, 45};
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S3(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 45 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_6]], %[[VAL_8]], %[[VAL_10]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.alloca !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_12]] : !cc.ptr<i1>
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_13]] : !cc.ptr<i1>
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_11]][2] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_14]] : !cc.ptr<i1>
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_11]][3] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_15]] : !cc.ptr<i32>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           return %[[VAL_16]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:         }
// clang-format on

struct S4 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    auto result = help_me_help_you();
    result.x = 47;
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S4(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 47 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_2]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_4]], %[[VAL_6]], %[[VAL_8]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_9:.*]] = call @_Z16help_me_help_youv() : () -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][3] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_10]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           return %[[VAL_12]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:         }
// clang-format on

struct S5 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    ResultThing result;
    result.m1 = (arg.i > 2);
    result.m2_ok = (arg.j > 4);
    result.m2 = arg.d != 0.0;
    result.x = arg.i - arg.j;
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S5(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_6]], %[[VAL_8]], %[[VAL_10]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.alloca !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK-NOT:       call @_ZN11ResultThingC1Ev
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_11]] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_13:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = arith.cmpi sgt, %[[VAL_14]], %[[VAL_3]] : i32
// CHECK:           cc.store %[[VAL_15]], %[[VAL_12]] : !cc.ptr<i1>
// CHECK:           %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_17:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = cc.load %[[VAL_17]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = arith.cmpi sgt, %[[VAL_18]], %[[VAL_2]] : i32
// CHECK:           cc.store %[[VAL_19]], %[[VAL_16]] : !cc.ptr<i1>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_11]][2] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_22:.*]] = cc.load %[[VAL_21]] : !cc.ptr<f64>
// CHECK:           %[[VAL_23:.*]] = arith.cmpf one, %[[VAL_22]], %[[VAL_1]] : f64
// CHECK:           cc.store %[[VAL_23]], %[[VAL_20]] : !cc.ptr<i1>
// CHECK:           %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_11]][3] : (!cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_28:.*]] = cc.load %[[VAL_27]] : !cc.ptr<i32>
// CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_26]], %[[VAL_28]] : i32
// CHECK:           cc.store %[[VAL_29]], %[[VAL_24]] : !cc.ptr<i32>
// CHECK:           %[[VAL_30:.*]] = cc.load %[[VAL_11]] : !cc.ptr<!cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>>
// CHECK:           return %[[VAL_30]] : !cc.struct<"ResultThing" {i1, i1, i1, i32} [64,4]>
// CHECK:         }
// clang-format on

struct S6 {
  typedef struct {
    bool n1;
    int y;
    short n2;
    int n3;
  } T;

  T operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    T result;
    result.n1 = (arg.i > 2);
    result.n2 = (arg.j > 4);
    result.n3 = arg.d;
    result.y = arg.i - arg.j;
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S6(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>) -> !cc.struct<{i1, i32, i16, i32} [128,4]>
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_5]], %[[VAL_7]], %[[VAL_9]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.struct<{i1, i32, i16, i32} [128,4]>
// CHECK-NOT:       call @_ZN2S61TC1Ev
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.struct<{i1, i32, i16, i32} [128,4]>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[VAL_2]] : i32
// CHECK:           cc.store %[[VAL_14]], %[[VAL_11]] : !cc.ptr<i1>
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_10]][2] : (!cc.ptr<!cc.struct<{i1, i32, i16, i32} [128,4]>>) -> !cc.ptr<i16>
// CHECK:           %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = cc.load %[[VAL_16]] : !cc.ptr<i32>
// CHECK:           %[[VAL_18:.*]] = arith.cmpi sgt, %[[VAL_17]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_19:.*]] = cc.cast unsigned %[[VAL_18]] : (i1) -> i16
// CHECK:           cc.store %[[VAL_19]], %[[VAL_15]] : !cc.ptr<i16>
// CHECK:           %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_10]][3] : (!cc.ptr<!cc.struct<{i1, i32, i16, i32} [128,4]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_22:.*]] = cc.load %[[VAL_21]] : !cc.ptr<f64>
// CHECK:           %[[VAL_23:.*]] = cc.cast signed %[[VAL_22]] : (f64) -> i32
// CHECK:           cc.store %[[VAL_23]], %[[VAL_20]] : !cc.ptr<i32>
// CHECK:           %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_10]][1] : (!cc.ptr<!cc.struct<{i1, i32, i16, i32} [128,4]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64} [128,8]>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_28:.*]] = cc.load %[[VAL_27]] : !cc.ptr<i32>
// CHECK:           %[[VAL_29:.*]] = arith.subi %[[VAL_26]], %[[VAL_28]] : i32
// CHECK:           cc.store %[[VAL_29]], %[[VAL_24]] : !cc.ptr<i32>
// CHECK:           %[[VAL_30:.*]] = cc.load %[[VAL_10]] : !cc.ptr<!cc.struct<{i1, i32, i16, i32} [128,4]>>
// CHECK:           return %[[VAL_30]] : !cc.struct<{i1, i32, i16, i32} [128,4]>
// CHECK:         }
// clang-format on

