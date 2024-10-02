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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernel4test(
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.struq<!quake.veq<?>, !quake.veq<?>>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = quake.get_member %[[VAL_0]][0] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_12:.*]] = quake.get_member %[[VAL_0]][1] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_13:.*]] = quake.veq_size %[[VAL_12]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_21:.*]] = quake.get_member %[[VAL_0]][0] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_22:.*]] = quake.extract_ref %[[VAL_21]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_applyH._Z6applyHRN5cudaq5quditILm2EEE(%[[VAL_22]]) : (!quake.ref) -> ()
// CHECK:           %[[VAL_23:.*]] = quake.get_member %[[VAL_0]][1] : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> !quake.veq<?>
// CHECK:           %[[VAL_24:.*]] = quake.extract_ref %[[VAL_23]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_applyX._Z6applyXRN5cudaq5quditILm2EEE(%[[VAL_24]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void entry_initlist() {
  cudaq::qvector q(2), r(2);
  test tt{q, r};
  kernel(tt);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry_initlist._Z14entry_initlistv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.make_struq %[[VAL_0]], %[[VAL_1]] : (!quake.veq<2>, !quake.veq<2>) -> !quake.struq<!quake.veq<?>, !quake.veq<?>>
// CHECK:           call @__nvqpp__mlirgen__function_kernel._Z6kernel4test(%[[VAL_2]]) : (!quake.struq<!quake.veq<?>, !quake.veq<?>>) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void entry_ctor() {
  cudaq::qvector q(2), r(2);
  test tt(q, r);
  h(tt.r[0]);
}

// clang-format off

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry_ctor._Z10entry_ctorv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_1]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// clang-format on
