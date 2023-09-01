/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector<double> initializer_list constructor support

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void testDouble() {
  cudaq::qubit q;
  std::vector<double> angle{M_PI_2, M_PI_4, 2*M_PI/40.};
  ry(angle[0], q);
  ry(angle[1], q);
  ry(angle[2], q);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testDouble._Z10testDoublev() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 0.15707963267948966 : f64
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 0.78539816339744828 : f64
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<f64 x 3>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_8]][0] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_10]]) %[[VAL_3]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_13]]) %[[VAL_3]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_14:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<f64 x 3>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_14]][2] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_16]]) %[[VAL_3]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }