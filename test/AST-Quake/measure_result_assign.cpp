/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// XFAIL: *
// TODO: Assignment to measurements collection elements (results[0] = mz(...))
// requires a set_measure op or equivalent to support mutable access into
// !quake.measurements<N>. Currently get_measure returns a value, not a pointer.

#include <cudaq.h>

__qpu__ bool assign_kernel() {
  cudaq::qvector q(2);
  auto results = mz(q);
  results[0] = mz(q[1]);
  return static_cast<bool>(results[0]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_assign_kernel._Z13assign_kernelv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "results" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
// CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>>
// CHECK:           %[[VAL_3:.*]] = cc.cast %[[VAL_2]] : (!cc.ptr<!cc.array<!quake.measure x ?>>) -> !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_0]][1] : (!quake.veq<2>) -> !quake.ref
// CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_4]] : (!quake.ref) -> !quake.measure
// CHECK:           cc.store %[[VAL_5]], %[[VAL_3]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<!quake.measure>) -> !cc.ptr<!cc.array<!quake.measure x ?>>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<!cc.array<!quake.measure x ?>>) -> !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<!quake.measure>
// CHECK:           %[[VAL_9:.*]] = quake.discriminate %[[VAL_8]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_9]] : i1
// CHECK:         }
// clang-format on
