/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void f() {
   cudaq::qvector v = {1.0, 2.0, 3.0, 4.0};
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_f._Z1fv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = cc.address_of @__nvqpp__rodata_init_0 : !cc.ptr<!cc.array<f64 x 4>>
// CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_2:.*]] = quake.init_state %[[VAL_1]], %[[VAL_0]] : (!quake.veq<2>, !cc.ptr<!cc.array<f64 x 4>>) -> !quake.veq<2>
// CHECK:           return
// CHECK:         }

__qpu__ void g() {
   cudaq::qvector v;
   std::vector<double> dv = {5.0, 6.0, 7.0, 8.0};
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_g._Z1gv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 8.000000e+00 : f64
// CHECK:           %[[VAL_1:.*]] = arith.constant 7.000000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = arith.constant 6.000000e+00 : f64
// CHECK:           %[[VAL_3:.*]] = arith.constant 5.000000e+00 : f64
// CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<1>
// CHECK:           %[[VAL_5:.*]] = cc.alloca !cc.array<f64 x 4>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_5]][1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_7]] : !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_5]][2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_8]] : !cc.ptr<f64>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_5]][3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           return
// CHECK:         }

// CHECK:         cc.global constant @__nvqpp__rodata_init_0 (dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf64>) : !cc.array<f64 x 4>

