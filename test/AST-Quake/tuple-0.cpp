/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct ArithmeticTupleQernel {
  void operator()(std::tuple<int, double, short, long> t) __qpu__ {
    cudaq::qvector q(1);
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticTupleQernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{i32, f64, i16, i64}>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<{i32, f64, i16, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<{i32, f64, i16, i64}>>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

struct ArithmeticPairQernel {
  void operator()(std::pair<float, int> t) __qpu__ {
    cudaq::qvector q(1);
    mz(q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticPairQernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{f32, i32}>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<{f32, i32}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<{f32, i32}>>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
