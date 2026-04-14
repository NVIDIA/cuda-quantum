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
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<20>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<20>) -> !quake.measurements<20>
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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq() -> !quake.measurements<?> 
// CHECK-NOT: cudaq-entrypoint
// CHECK-DAG:       %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.veq<4>
// CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : (!quake.ref, !quake.veq<4>, !quake.veq<2>, !quake.ref) -> !quake.measurements<8>
// CHECK:           %[[VAL_5:.*]] = quake.relax_size %[[VAL_4]] : (!quake.measurements<8>) -> !quake.measurements<?>
// CHECK:           return %[[VAL_5]] : !quake.measurements<?>
// CHECK:         }
// clang-format on

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

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfStaticVeq_Bool() -> !cc.stdvec<i1>
// CHECK-SAME:      attributes {"cudaq-entrypoint"
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]], %[[VAL_2]], %[[VAL_3]], %[[VAL_4]] name "res" : (!quake.ref, !quake.veq<4>, !quake.veq<2>, !quake.ref) -> !quake.measurements<8>
// CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!quake.measurements<8>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_6]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.stdvec_size %[[VAL_6]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_9:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_7]], %[[VAL_8]], %[[VAL_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_10:.*]] = cc.stdvec_init %[[VAL_9]], %[[VAL_8]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_10]] : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct VectorOfDynamicVeq {
  std::vector<cudaq::measure_result> operator()(unsigned i, unsigned j) __qpu__ {
    cudaq::qubit q1;
    cudaq::qvector reg1(i);
    cudaq::qvector reg2(j);
    cudaq::qubit q2;
    return mz(q1, reg1, reg2, q2);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorOfDynamicVeq(
// CHECK-SAME:                                                    %[[VAL_0:.*]]: i32,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: i32) -> !quake.measurements<?> attributes {"cudaq-kernel"} {
// CHECK-NOT: cudaq-entrypoint
// CHECK:           %[[VAL_2:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_2]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.cast unsigned %[[VAL_5]] : (i32) -> i64
// CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_6]] : i64]
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.cast unsigned %[[VAL_8]] : (i32) -> i64
// CHECK:           %[[VAL_10:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_9]] : i64]
// CHECK:           %[[VAL_11:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_12:.*]] = quake.mz %[[VAL_4]], %[[VAL_7]], %[[VAL_10]], %[[VAL_11]] : (!quake.ref, !quake.veq<?>, !quake.veq<?>, !quake.ref) -> !quake.measurements<?>
// CHECK:           return %[[VAL_12]] : !quake.measurements<?>
// CHECK:         }
// clang-format on

struct MxTest {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto r = mx(q);
    bool b = r;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MxTest() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.mx %[[VAL_0]] name "r" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_3:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           return
// CHECK:         }
// clang-format on

struct MyTest {
  void operator()() __qpu__ {
    cudaq::qubit q;
    auto r = my(q);
    bool b = r;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__MyTest() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = quake.my %[[VAL_0]] name "r" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_3:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_2]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK:           return
// CHECK:         }
// clang-format on
