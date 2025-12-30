/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void qview_test(cudaq::qview<> v) {}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qview_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>)

__qpu__ void qvector_test(cudaq::qvector<> v) {}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qvector_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>)

__qpu__ void qarray_test(cudaq::qarray<4> a) {}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_qarray_test
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<4>)

struct Qernel0 {
  void operator()() __qpu__ {
    cudaq::qvector bits(2);
    cudaq::qview scenicview = {bits};
    mz(scenicview);
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Qernel0()
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }
// clang-format on
