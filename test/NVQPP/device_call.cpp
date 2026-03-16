/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This file check AOT device call lowering for device functions that only have
// declarations (i.e., no definitions) in the module. For remote hardware
// providers, this is supported as the implementation is provided by the
// provider.
// For local execution targets, this is not supported as we have no way to
// resolve the device call for simulation or emulation.
// These should fail to compile because the device call cannot be resolved for
// local execution targets.
// clang-format off
// RUN: nvq++ %s -o %t 2>&1 | FileCheck %s --check-prefix=COMPILE_ERROR
// RUN: nvq++ %s -o --target quantinuum --emulate %t 2>&1 | FileCheck %s --check-prefix=COMPILE_ERROR
// clang-format on

// This should compile successfully because the trap implementation will be
// inserted for unresolved device calls.

// RUN: nvq++ --target quantinuum %s -o %t

#include "cudaq.h"

extern "C" {
// Device call declaration
int add_op(int a, int b);
}

__qpu__ int kernel(int a, int b) {
  int result = cudaq::device_call(/*device_id*/ 0, add_op, a, b);
  return result;
}

int main() {
  int a = 2;
  int b = 3;
  auto results = cudaq::run(100, kernel, a, b);
  return 0;
}

// COMPILE_ERROR: undefined reference to `add_op'
