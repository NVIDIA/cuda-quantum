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
  auto kernel = [](double theta) __qpu__ {
    cudaq::qreg q(4);
    x(q[0]);
    x(q[1]);
    exp_pauli(theta, q, "XXXY");
  };
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__Z4mainE3$_0(
// CHECK-SAME:                                             %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_2]][0] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_2]][1] : (!quake.veq<4>) -> !quake.ref
// CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_1]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.string_literal "XXXY" : !cc.ptr<!cc.array<i8 x 5>>
// CHECK:           quake.exp_pauli(%[[VAL_5]]) %[[VAL_2]], %[[VAL_6]] : (f64, !quake.veq<4>, !cc.ptr<!cc.array<i8 x 5>>) -> ()
// CHECK:           return
// CHECK:         }
