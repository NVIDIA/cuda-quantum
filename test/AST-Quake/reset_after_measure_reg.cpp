/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --indirect-to-direct-calls --inline --expand-measurements --memtoreg=quantum=0 --canonicalize --cc-loop-normalize --cc-loop-unroll --qubit-reset-before-reuse | FileCheck %s
// clang-format on

#include <cudaq.h>

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_reuse1
void reuse1() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  auto res = mz(q);
  // Measure is expanded to measure + reset + initialization
  // clang-format off
  // q[0]
  // CHECK: %[[RES_0:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT_0:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure 
  // CHECK-NEXT: quake.reset %[[QUBIT_0]] : (!quake.ref) -> () 
  // CHECK-NEXT: %[[BIT_0:[0-9]+]] = quake.discriminate %[[RES_0]] : (!quake.measure) -> i1 
  // CHECK-NEXT: cc.if(%[[BIT_0]]) { 
  // CHECK-NEXT: quake.x %[[QUBIT_0]] : (!quake.ref) -> () 
  // CHECK-NEXT:  }
  // q[1]
  // CHECK: %[[RES_1:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT_1:[0-9]+]] name "res" : (!quake.ref) -> !quake.measure 
  // CHECK-NEXT: quake.reset %[[QUBIT_1]] : (!quake.ref) -> () 
  // CHECK-NEXT: %[[BIT_1:[0-9]+]] = quake.discriminate %[[RES_1]] : (!quake.measure) -> i1 
  // CHECK-NEXT: cc.if(%[[BIT_1]]) { 
  // CHECK-NEXT: quake.x %[[QUBIT_1]] : (!quake.ref) -> () 
  // CHECK-NEXT:  }
  // clang-format on
  if (res[0]) {
    h(q);
  }
}

void foo(cudaq::qview<> q) __qpu__ { h(q); }

// Call other kernels
// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_reuse2
void reuse2() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
  // Measure is expanded to measure + reset + initialization
  // clang-format off
  // q[0]
  // CHECK: %[[RES_0:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT_0:[0-9]+]] : (!quake.ref) -> !quake.measure 
  // CHECK-NEXT: quake.reset %[[QUBIT_0]] : (!quake.ref) -> () 
  // CHECK-NEXT: %[[BIT_0:[0-9]+]] = quake.discriminate %[[RES_0]] : (!quake.measure) -> i1 
  // CHECK-NEXT: cc.if(%[[BIT_0]]) { 
  // CHECK-NEXT: quake.x %[[QUBIT_0]] : (!quake.ref) -> () 
  // CHECK-NEXT:  }
  // q[1]
  // CHECK: %[[RES_1:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT_1:[0-9]+]] : (!quake.ref) -> !quake.measure 
  // CHECK-NEXT: quake.reset %[[QUBIT_1]] : (!quake.ref) -> () 
  // CHECK-NEXT: %[[BIT_1:[0-9]+]] = quake.discriminate %[[RES_1]] : (!quake.measure) -> i1 
  // CHECK-NEXT: cc.if(%[[BIT_1]]) { 
  // CHECK-NEXT: quake.x %[[QUBIT_1]] : (!quake.ref) -> () 
  // CHECK-NEXT:  }
  // clang-format on
  // Call other kernel, which is inlined.
  foo(q);
}

// Use q[1] only
void bar(cudaq::qview<> q) __qpu__ { h(q[1]); }

// CHECK-LABEL: func.func @__nvqpp__mlirgen__function_reuse3
void reuse3() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  mz(q);
  // clang-format off
  // ========================================
  // First measure on q[0], no reset is needed as it is not reused
  // CHECK: quake.mz  
  // CHECK-NOT: quake.reset
  // ========================================
  // Measure on q[1] -> reset appended
  // CHECK: %[[RES:[a-zA-Z0-9_]+]] = quake.mz %[[QUBIT:[0-9]+]] : (!quake.ref) -> !quake.measure 
  // CHECK-NEXT: quake.reset %[[QUBIT]] : (!quake.ref) -> () 
  // CHECK-NEXT: %[[BIT:[0-9]+]] = quake.discriminate %[[RES]] : (!quake.measure) -> i1 
  // CHECK-NEXT: cc.if(%[[BIT]]) { 
  // CHECK-NEXT: quake.x %[[QUBIT]] : (!quake.ref) -> () 
  // CHECK-NEXT:  }
  // clang-format on
  // Call other kernel, which is inlined.
  bar(q);
}
