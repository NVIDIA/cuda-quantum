/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s 2>&1 | FileCheck %s

#include <cudaq.h>

int main() {

  auto [kernel, value] = cudaq::make_kernel<float>();
  auto q = kernel.qalloc();
  kernel.x(q);

  // Calling sample but not passing along a concrete argument for `value`
  auto result = cudaq::sample(kernel);
}

// CHECK: 'InvalidArgs<cudaq::Msg<{{[0-9]+}}>{"requires <float>, got <>"}>'
