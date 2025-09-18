/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt --qubit-reset-before-reuse | FileCheck %s

#include <cudaq.h>
// CHECK:   func.func @__nvqpp__mlirgen__function_no_reuse
void no_reuse() __qpu__ {
  cudaq::qubit q;
  // CHECK: quake.h
  h(q);
  // CHECK-NEXT: quake.mz
  mz(q);
  // CHECK-NEXT: return
}

// CHECK:   func.func @__nvqpp__mlirgen__function_explicit_reset_after_mz
void explicit_reset_after_mz() __qpu__ {
  cudaq::qubit q;
  // CHECK: quake.h
  h(q);
  // CHECK-NEXT: quake.mz
  mz(q);
  // CHECK-NEXT: quake.reset
  reset(q); // Explicit reset
  // CHECK-NEXT: quake.x
  x(q); // Reuse
  // CHECK-NEXT: return
}

// CHECK:   func.func @__nvqpp__mlirgen__function_reuse1
void reuse1() __qpu__ {
  cudaq::qubit q;
  // CHECK: quake.h
  h(q);
  mz(q);
  // clang-format off
  // CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] : (!quake.ref) -> !quake.measure
  // CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
  // CHECK-NEXT: cc.if(%[[BIT]]) {
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT:  }
  // CHECK-NEXT: quake.h %[[QUBIT]] : (!quake.ref) -> ()
  // clang-format on
  h(q); // Reuse
  // CHECK-NEXT: return
}

// CHECK:   func.func @__nvqpp__mlirgen__function_reuse2
void reuse2() __qpu__ {
  cudaq::qubit q;
  // CHECK: quake.h
  h(q);
  auto res = mz(q);
  // This is our automatic injection
  // clang-format off
  // CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure
  // CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
  // CHECK-NEXT: cc.if(%[[BIT]]) {
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT:  }
  // clang-format on

  // This reset-by-conditional-x pattern is also a reuse. Effectively, we now
  // perform a `reset - x - x` sequence == reset.
  // clang-format off
  // CHECK-NEXT: cc.alloca
  // CHECK-NEXT: cc.store
  // CHECK-NEXT: cc.load
  // CHECK-NEXT: cc.if
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT:  }
  // clang-format on
  if (res)
    x(q);
  // CHECK-NEXT: return
}

// CHECK:   func.func @__nvqpp__mlirgen__function_reuse3
void reuse3() __qpu__ {
  cudaq::qubit q, r;
  // CHECK: quake.h
  h(q);
  // CHECK-NEXT: quake.x [%[[QUBIT:[0-9]+]]] %[[QUBIT_1:[0-9]+]] : (!quake.ref,
  // !quake.ref) -> ()
  cx(q, r);
  auto res = mz(q);
  // This is our automatic injection
  // clang-format off
  // CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure
  // CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
  // CHECK-NEXT: cc.if(%[[BIT]]) {
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT:  }
  // clang-format on

  // This reset-by-conditional-x pattern is also a reuse. Effectively, we now
  // perform a `reset - x - x` sequence == reset.
  // clang-format off
  // CHECK-NEXT: cc.alloca
  // CHECK-NEXT: cc.store
  // CHECK-NEXT: cc.load
  // CHECK-NEXT: cc.if
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
  // CHECK-NEXT: quake.x %[[QUBIT_1]] : (!quake.ref) -> ()
  // CHECK-NEXT:  }
  // clang-format on
  if (res) {
    x(q);
    x(r); // r was not measured, hence no reset
  }

  // CHECK-NEXT: return
}
