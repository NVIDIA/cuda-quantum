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

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__t1(
// CHECK-SAME:        %[[VAL_0:.*]]: !cc.stdvec<f64>{{.*}}) -> i1 attributes
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "vec" : (!quake.veq<2>) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_3:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<i1>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<i1>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i1>
// CHECK:           return %[[VAL_5]] : i1
// CHECK:         }
// CHECK-NOT:     func.func private @_ZNKSt14_Bit_referencecvbEv() -> i1

