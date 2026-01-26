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
#include <vector>

struct ak1 {
  auto operator()(int i) __qpu__ {
    cudaq::qvector q(2);
    auto vec = mz(q);
    return vec[0]; 
  }
};

// CHECK: #[[$ATTR_0:.+]] = loc("auto_kernel-1.cpp":17:3)
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak1(
// CHECK-SAME:      %[[ARG0:.*]]: i32 loc("auto_kernel-1.cpp":17:3)) -> !quake.measure attributes {"cudaq-kernel"} {
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32 loc(#loc2)
// CHECK:           cc.store %[[ARG0]], %[[ALLOCA_0]] : !cc.ptr<i32> loc(#loc2)
// CHECK:           %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<2> loc(#loc3)
// CHECK:           %[[MZ_0:.*]] = quake.mz %[[ALLOCA_1]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure> loc(#loc4)
// CHECK:           %[[STDVEC_DATA_0:.*]] = cc.stdvec_data %[[MZ_0]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>> loc(#loc5)
// CHECK:           %[[CAST_0:.*]] = cc.cast %[[STDVEC_DATA_0]] : (!cc.ptr<!cc.array<!quake.measure x ?>>) -> !cc.ptr<!quake.measure> loc(#loc5)
// CHECK:           %[[LOAD_0:.*]] = cc.load %[[CAST_0]] : !cc.ptr<!quake.measure> loc(#loc5)
// CHECK:           return %[[LOAD_0]] : !quake.measure loc(#loc6)
// CHECK:         } loc(#[[$ATTR_0]])
