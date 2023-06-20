/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --pass-pipeline='builtin.module(func.func(memtoreg{quantum=0}),canonicalize,cc-loop-unroll,canonicalize)' | FileCheck %s

#include <cudaq.h>

struct test1 {
  void operator()(cudaq::qreg<> &q) __qpu__ {
    for (unsigned i = 0; i < 3; i += 1) {
      h(q[i]);
      z(q[i]);
      h(q[i]);
    }
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test1(
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct test2 {
  void operator()(cudaq::qreg<> &q) __qpu__ {
    for (unsigned i = 0; i <= 3; ++i) {
      h(q[i]);
      y(q[i]);
      h(q[i]);
    }
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test2(
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct test3 {
  void operator()(cudaq::qreg<> &q) __qpu__ {
     // Do not expect this to unroll. Loop must be normalized.
    for (unsigned i = 7; i < 14; i += 3) {
      h(q[i]);
      x(q[i]);
      h(q[i]);
    }
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test3(
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           return
