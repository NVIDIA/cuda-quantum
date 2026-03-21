/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
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
// CHECK:           %[[VAL_4:.*]] = cc.alloca !cc.array<!quake.measure x 3>
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_1]] name "result%[[VAL_0]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<!quake.measure x 3>>) -> !cc.ptr<!quake.measure>
// CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_7:.*]] = quake.mz %[[VAL_2]] name "result%[[VAL_1]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<!quake.measure x 3>>) -> !cc.ptr<!quake.measure>
// CHECK:           cc.store %[[VAL_7]], %[[VAL_8]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_9:.*]] = quake.mz %[[VAL_3]] name "result%[[VAL_2]]" : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_4]][2] : (!cc.ptr<!cc.array<!quake.measure x 3>>) -> !cc.ptr<!quake.measure>
// CHECK:           cc.store %[[VAL_9]], %[[VAL_10]] : !cc.ptr<!quake.measure>
// CHECK:           return
// CHECK:         }
