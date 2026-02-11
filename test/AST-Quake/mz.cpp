/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct S {
  void operator()() __qpu__ {
    cudaq::qvector reg(20);
    mz(reg);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S() attributes
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<20>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<20>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on

struct VectorOfStaticVeq {
  std::vector<cudaq::measure_result> operator()() __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(4);
    cudaq::qvector reg2(2);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq() -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_0:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] : (!quake.ref, !quake.veq<4>, !quake.veq<2>, !quake.ref) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_data %[[VAL_5]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_5]] : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           %[[VAL_8:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_6]], %[[VAL_7]], %[[VAL_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_8]], %[[VAL_7]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!quake.measure>
// CHECK:           return %[[VAL_9]] : !cc.stdvec<!quake.measure>
// CHECK:         }

struct VectorOfStaticVeq_Bool {
  std::vector<bool> operator()() __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(4);
    cudaq::qvector reg2(2);
    cudaq::qubit q2;
    auto res = mz(q1, reg1, reg2, q2);
    return cudaq::to_bool_vector(res);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq_Bool() -> !cc.stdvec<i1> attributes
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] name "res" : (!quake.ref, !quake.veq<4>, !quake.veq<2>, !quake.ref) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_6]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_size %[[VAL_6]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_9:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_7]], %[[VAL_8]], %[[VAL_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_init %[[VAL_9]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_10]] : !cc.stdvec<i1>
// CHECK:         }

struct VectorOfDynamicVeq {
  std::vector<cudaq::measure_result> operator()(unsigned i, unsigned j) __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(i);
    cudaq::qvector reg2(j);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfDynamicVeq(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: i32) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_2:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast unsigned %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_7]] : i64]
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_10:.*]] = cc.cast unsigned %[[VAL_9]] : (i32) -> i64
// CHECK:           %[[VAL_11:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_10]] : i64]
// CHECK:           %[[VAL_12:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_13:.*]] = quake.mz %[[VAL_5]], %[[VAL_8]], %[[VAL_11]], %[[VAL_12]] : (!quake.ref, !quake.veq<?>, !quake.veq<?>, !quake.ref) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_14:.*]] = cc.stdvec_data %[[VAL_13]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_15:.*]] = cc.stdvec_size %[[VAL_13]] : (!cc.stdvec<!quake.measure>) -> i64
// CHECK:           %[[VAL_16:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_14]], %[[VAL_15]], %[[VAL_2]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_17:.*]] = cc.stdvec_init %[[VAL_16]], %[[VAL_15]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!quake.measure>
// CHECK:           return %[[VAL_17]] : !cc.stdvec<!quake.measure>
// CHECK:         }
