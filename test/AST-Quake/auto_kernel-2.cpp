/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Simple test using a type inferenced return value type.

#include <cudaq.h>

struct ak2 {
  auto operator()() __qpu__ {
    cudaq::qarray<5> q;
    h(q);
    return mz(q);
  }
};

// CHECK: #[[$ATTR_0:.+]] = loc("auto_kernel-2.cpp":18:5)
// CHECK: #[[$ATTR_1:.+]] = loc("-":2:45)
// CHECK: #[[$ATTR_2:.+]] = loc("-":2:65)
// CHECK: #[[$ATTR_3:.+]] = loc("-":2:77)
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak2
// CHECK-SAME: () -> !cc.stdvec<!quake.measure> attributes {"cudaq-kernel"} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 5 : i64 loc(#[[$ATTR_0]])
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : i64 loc(#loc3)
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i64 loc(#[[$ATTR_0]])
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : i64 loc(#[[$ATTR_0]])
// CHECK:           %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5> loc(#loc4)
// CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_3]]) -> (i64)) {
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_0]] : i64 loc(#[[$ATTR_0]])
// CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]] : i64) loc(#[[$ATTR_0]])
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_1:.*]]: i64 loc("auto_kernel-2.cpp":18:5)):
// CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[VAL_1]]] : (!quake.veq<5>, i64) -> !quake.ref loc(#[[$ATTR_0]])
// CHECK:             quake.h %[[EXTRACT_REF_0]] : (!quake.ref) -> () loc(#[[$ATTR_0]])
// CHECK:             cc.continue %[[VAL_1]] : i64 loc(#[[$ATTR_0]])
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_2:.*]]: i64 loc("auto_kernel-2.cpp":18:5)):
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_2]] : i64 loc(#[[$ATTR_0]])
// CHECK:             cc.continue %[[ADDI_0]] : i64 loc(#[[$ATTR_0]])
// CHECK:           } {invariant} loc(#[[$ATTR_0]])
// CHECK:           %[[MZ_0:.*]] = quake.mz %[[ALLOCA_0]] : (!quake.veq<5>) -> !cc.stdvec<!quake.measure> loc(#loc5)
// CHECK:           %[[STDVEC_DATA_0:.*]] = cc.stdvec_data %[[MZ_0]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<i8> loc(#loc3)
// CHECK:           %[[STDVEC_SIZE_0:.*]] = cc.stdvec_size %[[MZ_0]] : (!cc.stdvec<!quake.measure>) -> i64 loc(#loc3)
// CHECK:           %[[VAL_3:.*]] = call @__nvqpp_vectorCopyCtor(%[[STDVEC_DATA_0]], %[[STDVEC_SIZE_0]], %[[CONSTANT_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8> loc(#loc3)
// CHECK:           %[[STDVEC_INIT_0:.*]] = cc.stdvec_init %[[VAL_3]], %[[STDVEC_SIZE_0]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!quake.measure> loc(#loc3)
// CHECK:           return %[[STDVEC_INIT_0]] : !cc.stdvec<!quake.measure> loc(#loc3)
// CHECK:         } loc(#loc1)
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1
// CHECK-LABEL: func.func private @__nvqpp_vectorCopyCtor(
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1

