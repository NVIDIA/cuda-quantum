/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include "cudaq.h"

// Value initialization of a scalar, `T()`, is zero.
struct ValueInit {
  double operator()() __qpu__ {
    int i = int();
    double d = double();
    return d + i;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__ValueInit(
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
// clang-format on
