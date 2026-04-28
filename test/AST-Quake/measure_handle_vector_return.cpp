/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// Pure-device kernels returning `std::vector<cudaq::measure_handle>`.
// Verifies the `ConvertStmt.cpp` heap-copy prologue
// (`cc.stdvec_data`/`cc.stdvec_size`/`__nvqpp_vectorCopyCtor`/
// `cc.stdvec_init`) fires for the handle element type, with element size
// 8 bytes from `getByteSizeOfType` -- the load-bearing case for QEC
// stabilizer rounds in CUDA-QX.

#include <cudaq.h>
#include <vector>

// Single qview: `mz(qv)` produces `!cc.stdvec<!cc.measure_handle>` and
// the return path heap-copies it with element size 8.
__qpu__ std::vector<cudaq::measure_handle>
single_round(cudaq::qview<> qv) {
  return mz(qv);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_single_round.
// CHECK-SAME:      -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[C8:.*]] = arith.constant 8 : i64
// CHECK:           %[[M:.*]] = quake.mz %{{.*}} : (!quake.veq<?>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[D:.*]] = cc.stdvec_data %[[M]] : (!cc.stdvec<!cc.measure_handle>) -> !cc.ptr<i8>
// CHECK:           %[[S:.*]] = cc.stdvec_size %[[M]] : (!cc.stdvec<!cc.measure_handle>) -> i64
// CHECK:           %[[H:.*]] = call @__nvqpp_vectorCopyCtor(%[[D]], %[[S]], %[[C8]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[V:.*]] = cc.stdvec_init %[[H]], %[[S]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           return %[[V]] : !cc.stdvec<!cc.measure_handle>

// Two qviews: matches the QEC stabilizer-round shape
// `mz(ancz, ancx)` from CUDA-QX `surface_code_device.cpp:77`.
__qpu__ std::vector<cudaq::measure_handle>
stab_round(cudaq::qview<> ancz, cudaq::qview<> ancx) {
  return mz(ancz, ancx);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_stab_round.
// CHECK-SAME:      -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %[[C8:.*]] = arith.constant 8 : i64
// CHECK:           %[[M:.*]] = quake.mz %{{.*}}, %{{.*}} : (!quake.veq<?>, !quake.veq<?>) -> !cc.stdvec<!cc.measure_handle>
// CHECK:           %{{.*}} = call @__nvqpp_vectorCopyCtor(%{{.*}}, %{{.*}}, %[[C8]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           return %{{.*}} : !cc.stdvec<!cc.measure_handle>
