/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt --expand-measurements --classical-optimization-pipeline | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ void foo() {
  cudaq::qvector qubits(3);
  x(qubits);
  auto result = mz(qubits);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_foo._Z3foov() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_1:.*]] = quake.extract_ref %[[VAL_0]][0] : (!quake.veq<3>) -> !quake.ref
// CHECK:           quake.x %[[VAL_1]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_2:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<3>) -> !quake.ref
// CHECK:           quake.x %[[VAL_2]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_3:.*]] = quake.extract_ref %[[VAL_0]][2] : (!quake.veq<3>) -> !quake.ref
// CHECK:           quake.x %[[VAL_3]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<i8 x 3>
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] name "result%[[VAL_0]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = cc.cast unsigned %[[VAL_6]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_8]], %[[VAL_7]] : !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = quake.mz %[[VAL_2]] name "result%[[VAL_1]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_10:.*]] = quake.discriminate %[[VAL_9]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_12:.*]] = cc.cast unsigned %[[VAL_10]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_12]], %[[VAL_11]] : !cc.ptr<i8>
// CHECK:           %[[VAL_13:.*]] = quake.mz %[[VAL_3]] name "result%[[VAL_2]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_14:.*]] = quake.discriminate %[[VAL_13]] : (!quake.measure) -> i1
// CHECK:           %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<i8 x 3>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_16:.*]] = cc.cast unsigned %[[VAL_14]] : (i1) -> i8
// CHECK:           cc.store %[[VAL_16]], %[[VAL_15]] : !cc.ptr<i8>
// CHECK:           return
// CHECK:         }
