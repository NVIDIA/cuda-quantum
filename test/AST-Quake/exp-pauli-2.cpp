/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

int main() {
  auto kernel2 = [](double theta, const char *pauli) __qpu__ {
    cudaq::qreg q(4);
    x(q[0]);
    x(q[1]);
    exp_pauli(theta, q, pauli);
  };
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0(
// CHECK-SAME:                                             %[[VAL_0:.*]]: f64,
// CHECK-SAME:                                             %[[VAL_1:.*]]: !cc.ptr<i8>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.x %[[VAL_5]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           quake.exp_pauli(%[[VAL_6]]) %[[VAL_3]], %[[VAL_1]] : (f64, !quake.veq<4>, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }

