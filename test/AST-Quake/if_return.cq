/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt --canonicalize | FileCheck %s

#include <cudaq.h>

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_c
// CHECK-SAME: () -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant true
// CHECK:           cc.if(%[[VAL_2]]) {
// CHECK:             cc.unwind_return %[[VAL_1]] : i32
// CHECK:           }
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

struct kernel_c {
  int operator()() __qpu__ {
    if (true) {
      return 1;
    }
    return 0;
  }
};
