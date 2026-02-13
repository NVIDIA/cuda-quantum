/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// We should have compile error if the callback name overload is used on target
// that don't support device calls with callback names.
// clang-format off
// RUN: nvq++ %s -o %t 2>&1 | FileCheck %s --check-prefix=COMPILE_CHECK
// RUN: nvq++ --target quantinuum %s -o %t 2>&1 | FileCheck %s --check-prefix=COMPILE_CHECK
// clang-format on

// Note: the below test, we mimic the AOT pipeline of a quantum device target
// that supports callback names, where the unresolved device call is replaced
// with a trap. However, since we don't have the actual device implementation
// (with a JIT pipeline) in place yet, we just check for the expected error
// message at runtime when the trap is executed.

// clang-format off
// RUN: nvq++ -DCUDAQ_QUANTUM_DEVICE -DCUDAQ_DEVICE_CALL_WITH_CALLBACK_NAME_SUPPORTED %s -o %t && %t 2>&1 | FileCheck %s --check-prefix=RUNTIME_CHECK
// clang-format on

#include "cudaq.h"

__qpu__ int kernel(int a, int b) {
  int result = cudaq::device_call<int>(/*device_id*/ 0, "add_op", a, b);
  return result;
}

int main() {
  int a = 2;
  int b = 3;
  auto results = cudaq::run(100, kernel, a, b);
  return 0;
}

// COMPILE_CHECK: device_call with callback name not supported in this target

// RUNTIME_CHECK: illegal execution of unreachable code
