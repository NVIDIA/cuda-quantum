/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --qubit-reset-before-reuse | FileCheck %s

#include <cudaq.h>
void no_reuse() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
}

// CHECK:   func.func @__nvqpp__mlirgen__function_no_reuse
// CHECK: quake.h
// CHECK-NEXT: quake.mz
// CHECK-NEXT: return

void explicit_reset_after_mz() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q); // Explicit reset
  x(q);     // Reuse
}

// CHECK:   func.func @__nvqpp__mlirgen__function_explicit_reset_after_mz
// CHECK: quake.h
// CHECK-NEXT: quake.mz
// CHECK-NEXT: quake.reset
// CHECK-NEXT: quake.x
// CHECK-NEXT: return

void reuse1() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);

  h(q); // Reuse
}

// clang-format off
// CHECK:   func.func @__nvqpp__mlirgen__function_reuse1
// CHECK: quake.h
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
// CHECK-NEXT: cc.if(%[[BIT]]) {
// CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT: quake.h %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT: return
// clang-format on

void reuse2() __qpu__ {
  cudaq::qubit q;
  h(q);
  auto res = mz(q);

  if (res)
    x(q);
}

// clang-format off
// CHECK:   func.func @__nvqpp__mlirgen__function_reuse2
// CHECK: quake.h
// This is our automatic injection
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
// CHECK-NEXT: cc.if(%[[BIT]]) {
// CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT:  }

// This reset-by-conditional-x pattern is also a reuse. Effectively, we now
// perform a `reset - x - x` sequence == reset.
// CHECK-NEXT: cc.alloca
// CHECK-NEXT: cc.store
// CHECK-NEXT: cc.load
// CHECK-NEXT: cc.if
// CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT:  }
// CHECK-NEXT: return
// clang-format on

void reuse3() __qpu__ {
  cudaq::qubit q, r;
  h(q);

  cx(q, r);
  auto res = mz(q);

  if (res) {
    x(q);
    x(r); // r was not measured, hence no reset
  }
}

// clang-format off
// CHECK:   func.func @__nvqpp__mlirgen__function_reuse3
// CHECK: quake.h
// CHECK-NEXT: quake.x [%[[QUBIT:[0-9]+]]] %[[QUBIT_1:[0-9]+]] : (!quake.ref, !quake.ref) -> ()
// This is our automatic injection
// CHECK-NEXT: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1
// CHECK-NEXT: cc.if(%[[BIT]]) {
// CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> ()
// CHECK-NEXT:  }

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
// CHECK-NEXT: return
// clang-format on
