/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <cmath>

// Standard math functions are usable inside kernels and lower to the matching
// MLIR math dialect ops. The angle is a runtime parameter so the calls are not
// constant-folded away.

struct MathFunctions {
  void operator()(double x) __qpu__ {
    cudaq::qubit q;
    rx(std::sin(x), q);
    rx(std::cos(x), q);
    rx(std::tan(x), q);
    rx(std::asin(x), q);
    rx(std::acos(x), q);
    rx(std::atan(x), q);
    rx(std::sqrt(x), q);
    rx(std::exp(x), q);
    rx(std::log(x), q);
  }
};

struct MathFunctionsFloat {
  void operator()(float x) __qpu__ {
    cudaq::qubit q;
    rx(sinf(x), q);
    rx(cosf(x), q);
    rx(tanf(x), q);
    rx(asinf(x), q);
    rx(acosf(x), q);
    rx(atanf(x), q);
    rx(sqrtf(x), q);
    rx(expf(x), q);
    rx(logf(x), q);
  }
};

// CHECK-LABEL: func.func @__nvqpp__mlirgen__MathFunctions
// CHECK-DAG: math.sin
// CHECK-DAG: math.cos
// CHECK-DAG: math.tan
// CHECK-DAG: math.asin
// CHECK-DAG: math.acos
// CHECK-DAG: math.atan
// CHECK-DAG: math.sqrt
// CHECK-DAG: math.exp
// CHECK-DAG: math.log
// CHECK: quake.rx

// CHECK-LABEL: func.func @__nvqpp__mlirgen__MathFunctionsFloat
// CHECK-DAG: math.sin
// CHECK-DAG: math.cos
// CHECK-DAG: math.tan
// CHECK-DAG: math.asin
// CHECK-DAG: math.acos
// CHECK-DAG: math.atan
// CHECK-DAG: math.sqrt
// CHECK-DAG: math.exp
// CHECK-DAG: math.log
// CHECK: quake.rx
