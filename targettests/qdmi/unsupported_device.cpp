/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: qdmi
// clang-format off
// RUN: nvq++ --target qdmi --qdmi-device mqt.sc.default %s -o %t
// RUN: not %t 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

struct empty_kernel {
  void operator()() __qpu__ {
    cudaq::qubit qubit;
    mz(qubit);
  }
};

int main() { static_cast<void>(cudaq::sample(empty_kernel{})); }

// CHECK: QDMI device supports none of CUDA-Q's transport formats
