/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

#include <vector>

void fillVector(std::vector<int> &out, int seed);
void readVector(const std::vector<int> &in, int seed);


// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_byRefVecKernel._Z14byRefVecKernelSt6vectorIiSaIiEES1_(
// CHECK-SAME:      %[[ARG0:.*]]: !cc.stdvec<i32>,
// CHECK-SAME:      %[[ARG1:.*]]: !cc.stdvec<i32>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 13 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 42 : i32
// CHECK:           %[[ALLOCA_0:.*]] = cc.alloca i32
// CHECK:           cc.store %[[CONSTANT_1]], %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           %[[LOAD_0:.*]] = cc.load %[[ALLOCA_0]] : !cc.ptr<i32>
// CHECK:           cc.device_call @_Z10fillVectorRSt6vectorIiSaIiEEi(%[[ARG0]], %[[LOAD_0]]) : (!cc.stdvec<i32>, i32) -> () {by_ref_vec_arg_indices = array<i64: 0>}
// CHECK:           %[[ALLOCA_1:.*]] = cc.alloca i32
// CHECK:           cc.store %[[CONSTANT_0]], %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:           %[[LOAD_1:.*]] = cc.load %[[ALLOCA_1]] : !cc.ptr<i32>
// CHECK:           cc.device_call @_Z10readVectorRKSt6vectorIiSaIiEEi(%[[ARG1]], %[[LOAD_1]]) : (!cc.stdvec<i32>, i32) -> ()
// CHECK:           return
// CHECK:         }
// CHECK:         func.func private @_Z10fillVectorRSt6vectorIiSaIiEEi(!cc.stdvec<i32>, i32) attributes {"cudaq-devicecall"}
// CHECK:         func.func private @_Z10readVectorRKSt6vectorIiSaIiEEi(!cc.stdvec<i32>, i32) attributes {"cudaq-devicecall"}

// CHECK-LABEL:   func.func @_Z14byRefVecKernelSt6vectorIiSaIiEES1_(
// CHECK-SAME:      %[[ARG0:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i32>, !cc.ptr<i32>, !cc.ptr<i32>}>>,
// CHECK-SAME:      %[[ARG1:.*]]: !cc.ptr<!cc.struct<{!cc.ptr<i32>, !cc.ptr<i32>, !cc.ptr<i32>}>>) attributes {no_this} {
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void byRefVecKernel(std::vector<int> out, std::vector<int> input) {
  cudaq::device_call(fillVector, out, 42);
  cudaq::device_call(readVector, input, 13);
}
