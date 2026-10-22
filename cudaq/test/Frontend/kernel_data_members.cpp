/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// The data members of a kernel class are the leading arguments of the kernel.
// Each is saved in a stack slot, and the kernel uses the slot. A write to a
// member only changes the slot.

struct Qernel {
  char c = 1;
  int i = 4;
  double d = 2.2;
  void operator()(int n) __qpu__ {
    cudaq::qvector q(n);
    i = i + 1;
    if (c > 0)
      rx(d * i, q[0]);
    mz(q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__Qernel(
// CHECK-SAME: %[[C:.*]]: i8{{.*}}, %[[I:.*]]: i32{{.*}}, %[[D:.*]]: f64{{.*}}, %[[N:.*]]: i32{{.*}}) attributes {{{.*}}"cudaq-member-args" = 3 : i64, "cudaq-member-offsets" = array<i64: 0, 4, 8>}
// CHECK-DAG: %[[CS:.*]] = cc.alloca i8
// CHECK-DAG: cc.store %[[C]], %[[CS]] : !cc.ptr<i8>
// CHECK-DAG: %[[IS:.*]] = cc.alloca i32
// CHECK-DAG: cc.store %[[I]], %[[IS]] : !cc.ptr<i32>
// CHECK-DAG: %[[DS:.*]] = cc.alloca f64
// CHECK-DAG: cc.store %[[D]], %[[DS]] : !cc.ptr<f64>
// CHECK: cc.store %{{.*}}, %[[IS]] : !cc.ptr<i32>
// CHECK: cc.load %[[DS]] : !cc.ptr<f64>

// The host function does not have the members as arguments.
// CHECK-LABEL: func.func @_ZN6QernelclEi(%{{.*}}: !cc.ptr<i8>{{.*}}, %{{.*}}: i32{{.*}}) {
// clang-format on
