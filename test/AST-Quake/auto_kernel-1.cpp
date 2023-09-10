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
#include <vector>

struct ak1 {
  auto operator()(int i) __qpu__ {
    cudaq::qreg q(2);
    auto vec = mz(q);
    return vec[0];
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ak1(
// CHECK-SAME:      %[[VAL_0:.*]]: i32{{.*}}) -> i1 attributes
// CHECK:           %[[VAL_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<i32>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] name "vec" : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i1>
// CHECK:           return %[[VAL_6]] : i1
