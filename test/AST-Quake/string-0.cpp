/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s
// XFAIL: *

// Test that a std::string argument can be passed to a kernel.

#include <cudaq.h>

struct KernelWithString {
  void operator()(double angle, std::string data) __qpu__ {
    cudaq::qvector q(2);
    exp_pauli(angle, data.data(), q[0]);
  }
};

void test0(double theta) {
  std::string pauliWord("XYZ");
  KernelWithString{}(theta, pauliWord);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__KernelWithString(
// CHECK-SAME:      %[[VAL_0:.*]]: f64{{.*}}, %[[VAL_1:.*]]: !cc.charspan{{.*}})
// CHECK:           %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.charspan) -> !cc.ptr<i8>
// CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_7:.*]] = quake.concat %[[VAL_6]] : (!quake.ref) -> !quake.veq<1>
// CHECK:           quake.exp_pauli %[[VAL_4]], %[[VAL_7]], %[[VAL_5]] : (f64, !quake.veq<1>, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }
