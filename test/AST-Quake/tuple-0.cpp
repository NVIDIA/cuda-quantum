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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticTupleQernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{i32, f64, i16, i64}>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<{i32, f64, i16, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<{i32, f64, i16, i64}>>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct ArithmeticPairQernel {
  void operator()(std::pair<float, int> t) __qpu__ {
    cudaq::qvector q(1);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticPairQernel(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{f32, i32}>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<{f32, i32}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<{f32, i32}>>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<1>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct ArithmeticTupleQernelWithUse {
  void operator()(std::tuple<int, double, short, long> t) __qpu__ {
    cudaq::qvector q(std::get<int>(t));
    h(q);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticTupleQernelWithUse(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{i32, f64, i16, i64}>)
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<{i32, f64, i16, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<!cc.struct<{i32, f64, i16, i64}>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<{i32, f64, i16, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.cast signed %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<?>[%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = quake.veq_size %[[VAL_7]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i64
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_7]][%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_12]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_15]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_16:.*]] = quake.mz %[[VAL_7]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct ArithmeticTupleQernelWithUse0 {
  void operator()(std::tuple<int, double, short, long> t) __qpu__ {
    cudaq::qvector q(std::get<0>(t));
    h(q);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticTupleQernelWithUse0(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{i32, f64, i16, i64}>)
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<{i32, f64, i16, i64}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<!cc.struct<{i32, f64, i16, i64}>>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.struct<{i32, f64, i16, i64}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.cast signed %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<?>[%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = quake.veq_size %[[VAL_7]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i64
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_7]][%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_12]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_15]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_16:.*]] = quake.mz %[[VAL_7]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct ArithmeticPairQernelWithUse {
  void operator()(std::pair<float, int> t) __qpu__ {
    cudaq::qvector q(t.second);
    mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ArithmeticPairQernelWithUse(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.struct<{f32, i32}>)
// CHECK:           %[[VAL_1:.*]] = cc.alloca !cc.struct<{f32, i32}>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<!cc.struct<{f32, i32}>>
// CHECK:           %[[VAL_2:.*]] = cc.compute_ptr %[[VAL_1]][1] : (!cc.ptr<!cc.struct<{f32, i32}>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.cast signed %[[VAL_3]] : (i32) -> i64
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>[%[[VAL_4]] : i64]
// CHECK:           %[[VAL_6:.*]] = quake.mz %[[VAL_5]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on
