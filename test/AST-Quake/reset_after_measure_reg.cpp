/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++20
// RUN: cudaq-quake %s | cudaq-opt --inline --constant-propagation --expand-measurements --unrolling-pipeline --qubit-reset-before-reuse | FileCheck %s

#include <cudaq.h>
void reuse1() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  cx(q[0], q[1]);
  auto res = mz(q);
  if (res[0]) {
    x(q);
  }
}

