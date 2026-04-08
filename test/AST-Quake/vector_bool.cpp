/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

// Simple test using a std::vector<bool> operator.

#include <cudaq.h>

struct t1 {
  bool operator()(std::vector<double> d) __qpu__ {
    cudaq::qvector q(2);
    auto vec = mz(q);
    return vec[0];
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__t1(
// CHECK-SAME:                                    %[[VAL_0:.*]]: !cc.stdvec<f64>) -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] name "vec" : (!quake.veq<2>) -> !quake.measurements<2>
// CHECK:           %[[VAL_4:.*]] = quake.get_measure %[[VAL_3]]{{\[}}%[[VAL_1]]] : (!quake.measurements<2>, i64) -> !quake.measure
// CHECK:           %[[VAL_5:.*]] = quake.discriminate %[[VAL_4]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_5]] : i1
// CHECK:         }
// CHECK-NOT:     func.func private @_ZNKSt14_Bit_referencecvbEv() -> i1
// clang-format on

struct VectorBoolReturn {
   std::vector<bool> operator()() __qpu__ {
    cudaq::qvector q(4);
    auto res = mz(q);
    return cudaq::to_bool_vector(res);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorBoolReturn() -> !cc.stdvec<i1>
// CHECK-SAME:      attributes {"cudaq-entrypoint"
// CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "res" : (!quake.veq<4>) -> !quake.measurements<4>
// CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!quake.measurements<4>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_6:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_4]], %[[VAL_5]], %[[VAL_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_init %[[VAL_6]], %[[VAL_5]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
// CHECK:           return %[[VAL_7]] : !cc.stdvec<i1>
// CHECK:         }
// clang-format on

struct VectorMeasureResult {
   std::vector<cudaq::measure_result> operator()() __qpu__ {
    cudaq::qvector q(4);
    return mz(q);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorMeasureResult() -> !quake.measurements<?>
// CHECK-NOT:     cudaq-entrypoint
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<4>) -> !quake.measurements<4>
// CHECK:           %[[VAL_2:.*]] = quake.relax_size %[[VAL_1]] : (!quake.measurements<4>) -> !quake.measurements<?>
// CHECK-NOT:       quake.discriminate
// CHECK:           return %[[VAL_2]] : !quake.measurements<?>
// CHECK:         }
// clang-format on
