/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
    cudaq::qreg<5> q;
    h(q);
    return mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak2
// CHECK-SAME: () -> !cc.stdvec<i1> attributes {
// CHECK:           %[[VAL_22:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_19:.*]] = quake.mz %{{.*}} : (!quake.veq<5>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_20:.*]] = cc.stdvec_data %[[VAL_19]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_21:.*]] = cc.stdvec_size %[[VAL_19]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_23:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_20]], %[[VAL_21]], %[[VAL_22]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_24:.*]] = cc.stdvec_init %[[VAL_23]], %[[VAL_21]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_24]] : !cc.stdvec<i1>
// CHECK:         }
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1
// CHECK-LABEL: func.func private @__nvqpp_vectorCopyCtor(
// CHECK-NOT:   func.func {{.*}} @_ZNKSt14_Bit_referencecvbEv() -> i1

