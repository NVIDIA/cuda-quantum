/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ cudaq::measure_result device_helper(cudaq::qubit &q) {
  h(q);
  return mz(q);
}

__qpu__ bool entry_kernel() {
  cudaq::qubit q;
  auto m = device_helper(q);
  return static_cast<bool>(m);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_device_helper
// CHECK-SAME:      (%[[VAL_0:.*]]: !quake.ref) -> !quake.measure attributes {"cudaq-kernel", no_this} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] : (!quake.ref) -> !quake.measure
// CHECK:           return %[[VAL_1]] : !quake.measure
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry_kernel._Z12entry_kernelv() -> i1 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
// CHECK:           %[[VAL_1:.*]] = call @__nvqpp__mlirgen__function_device_helper{{.*}}(%[[VAL_0]]) : (!quake.ref) -> !quake.measure
// CHECK:           %[[VAL_2:.*]] = quake.discriminate %[[VAL_1]] : (!quake.measure) -> i1
// CHECK:           return %[[VAL_2]] : i1
// CHECK:         }
// clang-format on
