/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | FileCheck %s

#include <cudaq.h>

struct testCast {
  void operator()() __qpu__ {
    cudaq::qubit q0, q1;
    h(q0);    
    double bit = mz(q0);
    // This tests implicit casting from double to bool
    if (bit)
      x(q1);
    mz(q1);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__testCast() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_1]] : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_10:.*]] = quake.discriminate %[[VAL_3]] :
// CHECK:           %[[VAL_4:.*]] = cc.cast unsigned %[[VAL_10]] : (i1) -> f64
// CHECK:           %[[VAL_5:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = arith.cmpf une, %[[VAL_6]], %[[VAL_0]] : f64
// CHECK:           cc.if(%[[VAL_7]]) {
// CHECK:             quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = quake.mz %[[VAL_2]] : (!quake.ref) -> !quake.measure
// CHECK:           return
// CHECK:         }
