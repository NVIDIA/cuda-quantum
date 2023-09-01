/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test for std::vector initializer_list constructor support

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void test() {
  cudaq::qubit q;
  std::vector<double> angle(2);   
  angle[0] = M_PI_2;
  angle[1] = M_PI_4;
  ry(angle[0], q);
  ry(angle[1], q);

}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test._Z4testv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 0.78539816339744828 : f64
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<f64 x 2>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<f64>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][1] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_8]][0] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_10]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<f64 x 2>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_11]][1] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_13:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f64>
// CHECK:           quake.ry (%[[VAL_13]]) %[[VAL_2]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }