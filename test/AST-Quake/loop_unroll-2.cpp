/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
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

struct test4 {
  // Use post-increment.
  void operator()() __qpu__ {
    cudaq::qreg reg(1);
    for (size_t i = 0; i < 3; i++) {
      h(reg[0]);
      y(reg[0]);
      h(reg[0]);
    }
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test4(
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct test5 {
  // Loop that decrements. Loop is not unrolled. It needs to be normalized.
  void operator()() __qpu__ {
    cudaq::qreg reg(1);
    for (size_t i = 3; i > 0; --i)
      x(reg[0]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test5(
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct test6 {
  // Loop that decrements. Loop is not unrolled. It needs to be normalized.
  void operator()() __qpu__ {
    cudaq::qreg reg(1);
    for (size_t i = 3; i-- > 0;)
      x(reg[0]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__test6(
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

//===----------------------------------------------------------------------===//
// The next 2 cases are negative tests. It is impossible to fully unroll a loop
// when the number of iterations is not statically determinable.

struct cannot_unroll_dynamic_iterations {
  void operator()(unsigned size) __qpu__ {
    cudaq::qreg reg(1);
    for (size_t i = 0; i < size; ++i)
      x(reg[0]);
    for (size_t i = 0; i < size; i++)
      y(reg[0]);
    for (size_t i = 0; i < size; i += 1)
      z(reg[0]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__cannot_unroll_dynamic_iterations(
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct cannot_unroll_dynamic_iterations_2 {
  void operator()(unsigned size) __qpu__ {
    cudaq::qreg reg(size);
    for (size_t i = 0; i < reg.size(); ++i)
      x(reg[i]);
    for (size_t i = 0; i < reg.size(); i++)
      y(reg[i]);
    for (size_t i = 0; i < reg.size(); i += 1)
      z(reg[i]);
    mz(reg);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__cannot_unroll_dynamic_iterations_2(
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           cc.loop while
// CHECK:           } do {
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

//===----------------------------------------------------------------------===//
// The next 2 cases are minor syntax variations that should not change the
// semantics of the loop structure.

struct for_loop_3 {
  void operator()() __qpu__ {
    cudaq::qreg reg(3);
    for (size_t i = 0; i < reg.size(); ++i)
      z(reg[i]);
    for (size_t i = 0; i < reg.size(); i++)
      y(reg[i]);
    for (size_t i = 0; i < reg.size(); i += 1)
      x(reg[i]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__for_loop_3(
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.x %{{.*}} : (!quake.ref) -> ()
// CHECK:           return

struct for_loop_4 {
  void operator()() __qpu__ {
    cudaq::qreg reg(3);
    for (size_t i = 0, n = reg.size(); i < n; ++i)
      y(reg[i]);
    for (size_t i = 0, n = reg.size(); i < n; i++)
      h(reg[i]);
    for (size_t i = 0, n = reg.size(); i < n; i += 1)
      z(reg[i]);
    mz(reg);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__for_loop_4(
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.y %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.h %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.h %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.h %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.h %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK-NOT:       quake.z %{{.*}} : (!quake.ref) -> ()
// CHECK:           return
