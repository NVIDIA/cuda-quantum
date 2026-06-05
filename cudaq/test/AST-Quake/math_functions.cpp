/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cmath>
#include <cudaq.h>

struct math_functions {
  void operator()(double theta) __qpu__ {
    cudaq::qubit q;
    double angle = std::sin(theta) + std::cos(theta) + std::tan(theta) +
                   std::asin(theta) + std::acos(theta) + std::atan(theta) +
                   std::sqrt(theta) + std::exp(theta) + std::log(theta);
    rx(angle, q);
  }
};

struct math_functions_float {
  void operator()(float theta) __qpu__ {
    cudaq::qubit q;
    float angle = sinf(theta) + cosf(theta) + tanf(theta) + asinf(theta) +
                  acosf(theta) + atanf(theta) + sqrtf(theta) + expf(theta) +
                  logf(theta);
    rx(angle, q);
  }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__math_functions
// CHECK-DAG:     math.sin
// CHECK-DAG:     math.cos
// CHECK-DAG:     math.tan
// CHECK-DAG:     math.asin
// CHECK-DAG:     math.acos
// CHECK-DAG:     math.atan
// CHECK-DAG:     math.sqrt
// CHECK-DAG:     math.exp
// CHECK-DAG:     math.log
// CHECK:         quake.rx

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__math_functions_float
// CHECK-DAG:     math.sin
// CHECK-DAG:     math.cos
// CHECK-DAG:     math.tan
// CHECK-DAG:     math.asin
// CHECK-DAG:     math.acos
// CHECK-DAG:     math.atan
// CHECK-DAG:     math.sqrt
// CHECK-DAG:     math.exp
// CHECK-DAG:     math.log
// CHECK:         quake.rx
