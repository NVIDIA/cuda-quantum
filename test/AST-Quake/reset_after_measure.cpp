/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_no_reuse._Z8no_reusev() 
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// clang-format on

void explicit_reset_after_mz() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);
  reset(q); // Explicit reset
  x(q);     // Reuse
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_explicit_reset_after_mz._Z23explicit_reset_after_mzv() 
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           quake.reset %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
// clang-format on

void reuse1() __qpu__ {
  cudaq::qubit q;
  h(q);
  mz(q);

  h(q); // Reuse
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_reuse1._Z6reuse1v() 
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           quake.reset %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           cc.if(%[[VAL_2]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// clang-format on

void reuse2() __qpu__ {
  cudaq::qubit q;
  h(q);
  auto res = mz(q);

  if (res)
    x(q);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_reuse2._Z6reuse2v() 
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "res" : (!quake.ref) -> !quake.measure
// CHECK:           quake.reset %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           cc.if(%[[VAL_2]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_3:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_4]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
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
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_reuse3._Z6reuse3v() 
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           quake.x {{\[}}%[[VAL_0]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_0]] name "res" : (!quake.ref) -> !quake.measure
// CHECK:           quake.reset %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!quake.measure) -> i1
// CHECK:           cc.if(%[[VAL_3]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_4:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_3]], %[[VAL_4]] : !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i1>
// CHECK:           cc.if(%[[VAL_5]]) {
// CHECK:             quake.x %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:             quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           }
// clang-format on
