/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-SIM %s
// RUN: nvq++ --target qci --emulate %s -o %t && %t 2>&1 | FileCheck --check-prefix=CHECK-EMUL %s
// clang-format on

#include <cudaq.h>
#include <cudaq/ptsbe/PTSBESample.h>
#include <iostream>

struct test_kernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    auto res = mz(q);
  }
};

__qpu__ bool test_run_kernel() {
  cudaq::qubit q;
  h(q);
  auto res = mz(q);
  return res;
}

int main() {
  auto counts = cudaq::sample(test_kernel{});
  counts.dump();

  std::cout << " before cudaq::run" << std::endl;
  auto run_results = cudaq::run(1000, test_run_kernel);

  std::cout << " before cudaq::ptsbe::sample" << std::endl;
  cudaq::ptsbe::sample_options options;
  options.shots = 100;
  options.ptsbe.return_execution_data = true;

  auto result = cudaq::ptsbe::sample(options, test_kernel{});
  return 0;
}

// clang-format off
// CHECK-SIM: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-SIM-NOT: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-SIM: before cudaq::run
// CHECK-SIM-NOT: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-SIM: before cudaq::ptsbe::sample
// CHECK-SIM: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-SIM-NOT: WARNING: Kernel "test_kernel" uses named measurement results
// clang-format on

// clang-format off
// CHECK-EMUL: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-EMUL-NOT: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-EMUL: before cudaq::run
// CHECK-EMUL-NOT: WARNING: Kernel "test_kernel" uses named measurement results
// CHECK-EMUL: before cudaq::ptsbe::sample
// clang-format on
