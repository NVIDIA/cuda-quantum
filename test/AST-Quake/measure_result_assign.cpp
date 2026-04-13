/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: not cudaq-quake %s 2>&1 | FileCheck %s

#include <cudaq.h>

__qpu__ bool assign_kernel() {
  cudaq::qvector q(2);
  auto results = mz(q);
  results[0] = mz(q[1]);
  return static_cast<bool>(results[0]);
}

// CHECK: error:{{.*}}deleted operator '='
// CHECK: error: C++ source has errors
