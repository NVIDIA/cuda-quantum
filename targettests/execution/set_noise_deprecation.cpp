/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

int main() {
  cudaq::noise_model noise;
  cudaq::set_noise(noise);
  cudaq::unset_noise();
  return 0;
}

// clang-format off
// CHECK: warning: 'set_noise' is deprecated: please use launch arguments or launch options.
// CHECK: warning: 'unset_noise' is deprecated: please use launch arguments or launch options.
// clang-format on
