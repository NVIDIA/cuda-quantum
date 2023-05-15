/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Simple test using a std::vector<bool> operator.

#include <cudaq.h>

struct t1 {
  bool operator()(std::vector<double> d) __qpu__ {
    cudaq::qreg q(2);
    auto vec = mz(q);
    return vec[0];
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__t1
// CHECK-SAME:        (%[[VAL_0:.*]]: !cc.stdvec<f64>) -> i1 attributes {{{.*}}"cudaq-entrypoint"{{.*}}} {
// CHECK:           %[[VAL_13:.*]] = quake.mz %{{.*}} : (!quake.veq<?>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_15:.*]] = arith.extsi %[[VAL_14]] : i32 to i64
// CHECK:           %[[VAL_16:.*]] = cc.stdvec_data %[[VAL_13]] : (!cc.stdvec<i1>) -> !llvm.ptr<i1>
// CHECK:           %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_16]][%[[VAL_15]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
// CHECK:           %[[VAL_18:.*]] = llvm.load %[[VAL_17]] : !llvm.ptr<i1>
// CHECK:           return %[[VAL_18]] : i1
// CHECK:         }
// CHECK-NOT:     func.func private @_ZNKSt14_Bit_referencecvbEv() -> i1

