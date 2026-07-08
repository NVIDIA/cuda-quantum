/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: if %qdmi_avail; then \
// RUN:   nvq++ --target qdmi --qdmi-library %qdmi_ddsim_library --qdmi-prefix %qdmi_ddsim_prefix %s -o %t && \
// RUN:   env DYLD_LIBRARY_PATH=%qdmi_ddsim_library_dir:$DYLD_LIBRARY_PATH LD_LIBRARY_PATH=%qdmi_ddsim_library_dir:$LD_LIBRARY_PATH %t | FileCheck %s; \
// RUN: fi
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  }
};

int main() {
  auto result = cudaq::sample(16, kernel{});
  std::cout << result.count("1") << '\n';
  return result.count("1") == 16 ? 0 : 1;
}

// CHECK: 16
