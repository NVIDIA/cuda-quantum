/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

#include <cudaq.h>

__qpu__ float magic_func(int i, int j, float f, float g) {
  int k = std::pow(i, j);
  float h = std::pow(f, k);
  return std::pow(g, h);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_magic_func._Z10magic_funciiff(
// CHECK:           %[[VAL_6:.*]] = math.ipowi %{{.*}}, %{{.*}} : i32
// CHECK:           %[[VAL_16:.*]] = math.fpowi %{{.*}}, %{{.*}} : f32, i32
// CHECK:           %[[VAL_20:.*]] = math.powf %{{.*}}, %{{.*}} : f32
// CHECK:         }

