/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ int rando_qernel(double);

__qpu__ void superstar_qernel(const cudaq::qkernel<int(double)>& bob, double dub) {
   auto size = bob(dub);
   cudaq::qvector q(size);
   mz(q);
}

void meanwhile_on_safari() {
   cudaq::qkernel<int(double)> tiger{rando_qernel};
   superstar_qernel(tiger, 11.0);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_superstar_qernel._Z16superstar_qernelRKN5cudaq7qkernelIFidEEEd(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.indirect_callable<(f64) -> i32>,
// CHECK-SAME:      %[[VAL_1:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_4:.*]] = cc.call_indirect_callable %[[VAL_0]], %[[VAL_3]] : (!cc.indirect_callable<(f64) -> i32>, f64) -> i32
// CHECK:           %[[VAL_5:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_4]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<?>[%[[VAL_7]] : i64]
// CHECK:           %[[VAL_9:.*]] = quake.mz %[[VAL_8]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_Z16superstar_qernelRKN5cudaq7qkernelIFidEEEd(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.ptr<i8>,
// CHECK-SAME:      %[[VAL_1:.*]]: f64) attributes {no_this} {
// CHECK:           return
// CHECK:         }
// clang-format on
