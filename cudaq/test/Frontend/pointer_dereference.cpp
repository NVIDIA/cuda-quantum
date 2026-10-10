/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include "cudaq.h"

// `*p` is an lvalue: the object that `p` points to. A pointer variable has
// storage of its own for the address (it is not the object that it points to),
// and the dereference is the address that is stored there. Previously the
// variable was an alias of the object, and a dereference asserted.

struct Scalar {
  int operator()() __qpu__ {
    int a = 1;
    int *p = &a;
    (*p)++;      // increment through the pointer
    *p = *p + 2; // assignment through the pointer
    *p += 3;     // compound assignment through the pointer
    return *p;   // read through the pointer
  }
};

struct P {
  int a;
  double b;
};

struct Aggregate {
  double operator()() __qpu__ {
    P s;
    s.b = 1.5;
    P *ps = &s;
    (*ps).a = 7;  // a member of the object that is pointed to
    ps->b *= 2.0; // and through the arrow
    return (*ps).b;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Scalar(
// CHECK-SAME:      ) -> i32
// CHECK:           %[[A:.*]] = cc.alloca i32
// CHECK:           cc.store %{{.*}}, %[[A]] : !cc.ptr<i32>
// The pointer variable has its own storage, which holds the address of `a`.
// CHECK:           %[[P:.*]] = cc.alloca !cc.ptr<i32>
// CHECK:           cc.store %[[A]], %[[P]] : !cc.ptr<!cc.ptr<i32>>
// (*p)++ loads the pointer, loads the value, adds one, stores through the pointer.
// CHECK:           %[[PTR0:.*]] = cc.load %[[P]] : !cc.ptr<!cc.ptr<i32>>
// CHECK:           %[[VAL0:.*]] = cc.load %[[PTR0]] : !cc.ptr<i32>
// CHECK:           %[[INC:.*]] = arith.addi %[[VAL0]], %{{.*}} : i32
// CHECK:           cc.store %[[INC]], %[[PTR0]] : !cc.ptr<i32>
// *p = *p + 2 stores through the loaded pointer.
// CHECK:           %[[PTR1:.*]] = cc.load %[[P]] : !cc.ptr<!cc.ptr<i32>>
// CHECK:           %[[PTR2:.*]] = cc.load %[[P]] : !cc.ptr<!cc.ptr<i32>>
// CHECK:           %[[VAL1:.*]] = cc.load %[[PTR2]] : !cc.ptr<i32>
// CHECK:           %[[SUM:.*]] = arith.addi %[[VAL1]], %{{.*}} : i32
// CHECK:           cc.store %[[SUM]], %[[PTR1]] : !cc.ptr<i32>
// *p += 3
// CHECK:           %[[PTR3:.*]] = cc.load %[[P]] : !cc.ptr<!cc.ptr<i32>>
// CHECK:           %[[VAL2:.*]] = cc.load %[[PTR3]] : !cc.ptr<i32>
// CHECK:           %[[SUM2:.*]] = arith.addi %[[VAL2]], %{{.*}} : i32
// CHECK:           cc.store %[[SUM2]], %[[PTR3]] : !cc.ptr<i32>
// return *p
// CHECK:           %[[PTR4:.*]] = cc.load %[[P]] : !cc.ptr<!cc.ptr<i32>>
// CHECK:           %[[RESULT:.*]] = cc.load %[[PTR4]] : !cc.ptr<i32>
// CHECK:           return %[[RESULT]] : i32

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Aggregate(
// CHECK-SAME:      ) -> f64
// CHECK:           %[[S:.*]] = cc.alloca !cc.struct<
// CHECK:           %[[PS:.*]] = cc.alloca !cc.ptr<!cc.struct<
// CHECK:           cc.store %[[S]], %[[PS]] : !cc.ptr<!cc.ptr<!cc.struct<
// (*ps).a = 7 stores to the first member of the object that is pointed to.
// CHECK:           %[[OBJ0:.*]] = cc.load %[[PS]] : !cc.ptr<!cc.ptr<!cc.struct<
// CHECK:           %[[MEMBER0:.*]] = cc.cast %[[OBJ0]] : (!cc.ptr<!cc.struct<{{.*}}>>) -> !cc.ptr<i32>
// CHECK:           cc.store %{{.*}}, %[[MEMBER0]] : !cc.ptr<i32>
// ps->b *= 2.0 updates the second member in place.
// CHECK:           %[[OBJ1:.*]] = cc.load %[[PS]] : !cc.ptr<!cc.ptr<!cc.struct<
// CHECK:           %[[MEMBER1:.*]] = cc.compute_ptr %[[OBJ1]][1] : (!cc.ptr<!cc.struct<{{.*}}>>) -> !cc.ptr<f64>
// CHECK:           %[[OLD:.*]] = cc.load %[[MEMBER1]] : !cc.ptr<f64>
// CHECK:           %[[NEW:.*]] = arith.mulf %[[OLD]], %{{.*}} : f64
// CHECK:           cc.store %[[NEW]], %[[MEMBER1]] : !cc.ptr<f64>
// clang-format on
