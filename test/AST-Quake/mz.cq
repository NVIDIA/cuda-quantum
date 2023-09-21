/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

struct S {
  void operator()() __qpu__ {
    cudaq::qreg reg(20);
    mz(reg);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S() attributes
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<20>
// CHECK:           %[[VAL_18:.*]] = quake.mz %[[VAL_2]] : (!quake.veq<20>) -> !cc.stdvec<i1>
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorOfStaticVeq {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qubit q1;
    cudaq::qreg reg1(4);
    cudaq::qreg reg2(2);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq() -> !cc.stdvec<i1> attributes {
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_0]], %[[VAL_3]], %[[VAL_6]], %[[VAL_7]] : (!quake.ref, !quake.veq<4>, !quake.veq<2>, !quake.ref) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_data %[[VAL_8]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_size %[[VAL_8]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_12:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_9]], %[[VAL_10]], %[[VAL_11]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = cc.stdvec_init %[[VAL_12]], %[[VAL_10]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_13]] : !cc.stdvec<i1>
// CHECK:         }

struct VectorOfDynamicVeq {
  std::vector<bool> operator()(unsigned i, unsigned j) __qpu__ {
    cudaq::qubit q1;
    cudaq::qreg reg1(i);
    cudaq::qreg reg2(j);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfDynamicVeq(
// CHECK-SAME:        %[[VAL_0:.*]]: i32{{.*}}, %[[VAL_1:.*]]: i32{{.*}}) -> !cc.stdvec<i1> attributes {
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = arith.extui %[[VAL_5]] : i32 to i64
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<?>[%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = arith.extui %[[VAL_8]] : i32 to i64
// CHECK:           %[[VAL_10:.*]] = quake.alloca !quake.veq<?>[%[[VAL_9]] : i64]
// CHECK:           %[[VAL_11:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_4]], %[[VAL_7]], %[[VAL_10]], %[[VAL_11]] : (!quake.ref, !quake.veq<?>, !quake.veq<?>, !quake.ref) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_12]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_14:.*]] = cc.stdvec_size %[[VAL_12]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_16:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_13]], %[[VAL_14]], %[[VAL_15]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = cc.stdvec_init %[[VAL_16]], %[[VAL_14]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_17]] : !cc.stdvec<i1>
// CHECK:         }

