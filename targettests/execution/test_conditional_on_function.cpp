/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t 2>&1 | FileCheck %s -check-prefix=FAIL

#include <cudaq.h>

__qpu__ auto measure(cudaq::qubit &r) { return mz(r); }

auto kernel_with_conditional_on_function = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  auto measureResult = measure(q[0]);
  if (measureResult)
    x(q[0]);
};

int main() {
  cudaq::sample_options options{.shots = 10, .explicit_measurements = true};
  cudaq::sample(options, kernel_with_conditional_on_function);
  return 0;
}

// FAIL: not supported on a kernel with conditional logic on a measurement result
