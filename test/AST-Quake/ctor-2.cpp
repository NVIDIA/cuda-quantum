/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s -o - | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct S1 {
  void operator()(bool b) __qpu__;
};

struct S2 {
  void operator()(bool b) __qpu__;
};

void S1::operator()(bool b) {
  cudaq::qubit q;
  S2 s2;
  s2(b);
  x(q);
}

void S2::operator()(bool b) {
  cudaq::qubit q;
  z(q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S1(
// CHECK-SAME:      %[[VAL_0:.*]]: i1)
// CHECK:           %[[VAL_1:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i1>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<"S2" {} [8,1]>
// CHECK-NOT:       call @_ZN2S2C1Ev
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_1]] : !cc.ptr<i1>
// CHECK:           call @_ZN2S2clEb(%[[VAL_4]]) : (i1) -> ()
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S2(
// CHECK-SAME:      %[[VAL_0:.*]]: i1)
// CHECK:           %[[VAL_1:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i1>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.z %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-NOT:     func.func private @_ZN2S2C1Ev
// CHECK:         func.func private @_ZN2S2clEb(i1)

// CHECK:         func.func @_ZN2S1clEb(
