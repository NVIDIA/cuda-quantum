/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S0(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<"ArgumentThing" {i32, i32, f64}>) -> !cc.struct<"ResultThing" {i1, i1, i1, i32}>
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<"ArgumentThing" {i32, i32, f64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64}>>
// CHECK:           %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][0] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_1]][1] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_1]][2] : (!cc.ptr<!cc.struct<"ArgumentThing" {i32, i32, f64}>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           call @_Z15debug_the_thingiid(%[[VAL_3]], %[[VAL_5]], %[[VAL_7]]) : (i32, i32, f64) -> ()
// CHECK:           %[[VAL_8:.*]] = call @_Z16help_me_help_youv() : () -> !cc.struct<"ResultThing" {i1, i1, i1, i32}>
// CHECK:           return %[[VAL_8]] : !cc.struct<"ResultThing" {i1, i1, i1, i32}>

#if 0
struct S1 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    ResultThing result = {true, false, false, 42};
    return result;
  }
};
#endif

#if 0
struct S2 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    return ResultThing(true, false, false, 42);
  }
};
#endif

#if 0
struct S3 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
    debug_the_thing(arg.i, arg.j, arg.d);
    return {true, false, false, 42};
  }
};
#endif

#if 0
struct S4 {
  ResultThing operator()(ArgumentThing arg) __qpu__ {
     ResultThing result;
     result.m1 = (arg.i > 2);
     result.m2_ok = (arg.j > 4);
     result.m2 = arg.d != 0.0;
     result.x = arg.i - arg.j;
    return result;
  }
};
#endif
