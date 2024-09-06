/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

struct test {
  cudaq::qview<> q;
  cudaq::qview<> r;
};

__qpu__ void applyH(cudaq::qubit &q) { h(q); }
__qpu__ void applyX(cudaq::qubit &q) { x(q); }
__qpu__ void kernel(test t) {
  h(t.q);
  s(t.r);

  applyH(t.q[0]);
  applyX(t.r[0]);
}

__qpu__ void entry_initlist() {
  cudaq::qvector q(2), r(2);
  test tt{q, r};
  kernel(tt);
}

// clang-format off

// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_3:.*]] = quake.relax_size %[[VAL_2]] : (!quake.veq<2>) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = cc.undef !cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_5:.*]] = cc.insert_value %[[VAL_1]], %[[VAL_4]][0] : (!cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>, !quake.veq<?>) -> !cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_6:.*]] = cc.insert_value %[[VAL_3]], %[[VAL_5]][1] : (!cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>, !quake.veq<?>) -> !cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>
// CHECK:           call @__nvqpp__mlirgen__function_kernel._Z6kernel4test(%[[VAL_6]]) : (!cc.struct<"test" {!quake.veq<?>, !quake.veq<?>} [256,8]>) -> ()
// CHECK:           return

// clang-format on

__qpu__ void entry_ctor() {
  cudaq::qvector q(2), r(2);
  test tt(q, r);
  h(tt.r[0]);
}

// clang-format off

// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           return

// clang-format on