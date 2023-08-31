/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

// CHECK: module attributes {quake.mangled_name_map = {__nvqpp__mlirgen__super{{.*}} = "_ZN5superclEd"}} {
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__super
// CHECK-SAME: (%[[arg0:.*]]: f64{{.*}}) -> i1
// CHECK:     %[[cst:.*]] = arith.constant 2.0{{.*}} : f64
// CHECK:     %[[V0:.*]] = cc.alloca f64
// CHECK:     cc.store %[[arg0]], %[[V0]] : !cc.ptr<f64>
// CHECK:     %[[V1:.*]] = quake.alloca !quake.ref
// CHECK:     %[[V2:.*]] = cc.load %[[V0]] : !cc.ptr<f64>
// CHECK:     quake.rx (%[[V2]]) %[[V1]] : (f64,
// CHECK:     %[[V3:.*]] = cc.load %[[V0]] : !cc.ptr<f64>
// CHECK:     %[[V4:.*]] = arith.divf %[[V3]], %[[cst]] : f64
// CHECK:     quake.ry (%[[V4]]) %[[V1]] : (f64,
// CHECK:     %[[V5:.*]] = quake.mz %[[V1]] : (!quake.ref) -> i1
// CHECK:     return %[[V5]] : i1
// CHECK:   }
// CHECK: }

struct super {
  bool operator()(double inputPi) __qpu__ {
    cudaq::qubit q;
    rx(inputPi, q);
    ry(inputPi / 2.0, q);
    return mz(q);
  }
};
