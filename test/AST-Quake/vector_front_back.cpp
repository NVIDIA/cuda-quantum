/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Simple tests of vector front/back on different std::vector<> types.

#include <cudaq.h>

__qpu__ void testFrontFloat() {
    std::vector<float> vec_float{0.0, 1.0, 2.0, 3.0};
    auto zero = vec_float.front();
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testFrontFloat._Z14testFrontFloatv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.array<f32 x 4>
// CHECK-DAG:       %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_7]] : !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_4]][3] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_8]] : !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_9:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_10:.*]] = cc.load %[[VAL_9]] : !cc.ptr<f32>
// CHECK-DAG:       %[[VAL_11:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_10]], %[[VAL_11]] : !cc.ptr<f32>
// CHECK:           return
// CHECK:         }


__qpu__ void testFrontBool() {
    std::vector<bool> vec_bool{0,1,0,1};
    bool zero = vec_bool.front();
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testFrontBool._Z13testFrontBoolv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant true
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant false
// CHECK-DAG:       %[[VAL_2:.*]] = cc.alloca !cc.array<i1 x 4>
// CHECK-DAG:       %[[VAL_3:.*]] = cc.compute_ptr %[[VAL_2]][0] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_3]] : !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][1] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_2]][2] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_2]][3] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_6]] : !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_7:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i1>
// CHECK-DAG:       %[[VAL_9:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_8]], %[[VAL_9]] : !cc.ptr<i1>
// CHECK:           return
// CHECK:         }

__qpu__ void testBackFloat() {
    std::vector<float> vec_float{0.0, 1.0, 2.0, 3.0};
    float three = vec_float.back();
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testBackFloat._Z13testBackFloatv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_3:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant -1 : i64
// CHECK:           %[[VAL_5:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_6:.*]] = cc.alloca !cc.array<f32 x 4>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][0] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_3]], %[[VAL_7]] : !cc.ptr<f32>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_6]][1] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_8]] : !cc.ptr<f32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_6]][2] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_9]] : !cc.ptr<f32>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_6]][3] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_10]] : !cc.ptr<f32>
// CHECK:           %[[VAL_11:.*]] = cc.stdvec_init %[[VAL_6]], %[[VAL_5]] : (!cc.ptr<!cc.array<f32 x 4>>, i64) -> !cc.stdvec<f32>
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<f32 x 4>>) -> !cc.ptr<f32>
// CHECK:           %[[VAL_13:.*]] = cc.stdvec_size %[[VAL_11]] : (!cc.stdvec<f32>) -> i64
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_14]]] : (!cc.ptr<f32>, i64) -> !cc.ptr<f32>
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f32>
// CHECK:           %[[VAL_17:.*]] = cc.alloca f32
// CHECK:           cc.store %[[VAL_16]], %[[VAL_17]] : !cc.ptr<f32>
// CHECK:           return
// CHECK:         }

__qpu__ void testBackBool() {
    std::vector<bool> vec_bool{0,1,0,1};
    bool one = vec_bool.back();
}


// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testBackBool._Z12testBackBoolv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = arith.constant true
// CHECK:           %[[VAL_1:.*]] = arith.constant false
// CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 4 : i64
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<i1 x 4>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_4]][0] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i1>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_6]] : !cc.ptr<i1>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_7]] : !cc.ptr<i1>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_4]][3] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_8]] : !cc.ptr<i1>
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_4]], %[[VAL_3]] : (!cc.ptr<!cc.array<i1 x 4>>, i64) -> !cc.stdvec<i1>
// CHECK:           %[[VAL_10:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i1 x 4>>) -> !cc.ptr<i1>
// CHECK:           %[[VAL_11:.*]] = cc.stdvec_size %[[VAL_9]] : (!cc.stdvec<i1>) -> i64
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : i64
// CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_12]]] : (!cc.ptr<i1>, i64) -> !cc.ptr<i1>
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i1>
// CHECK:           %[[VAL_15:.*]] = cc.alloca i1
// CHECK:           cc.store %[[VAL_14]], %[[VAL_15]] : !cc.ptr<i1>
// CHECK:           return
// CHECK:         }