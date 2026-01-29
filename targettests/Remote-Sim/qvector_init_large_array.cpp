/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// XFAIL: system-darwin
// ^^^^^ On macOS, the large array (4MB) is allocated on the stack via cc.alloca,
//       which exceeds the 512KB default stack size of asio worker threads used
//       by the cudaq-qpud HTTP server, causing a stack overflow.
//
//       This test is separated from qvector_init_from_vector.cpp to allow
//       regression testing of smaller arrays on all platforms.
//
//       Fixing this requires either:
//       1. Heap allocation for large arrays in codegen (ConvertExpr.cpp:3126)
//       2. Spawning JIT execution on a thread with larger stack size

// clang-format off
// RUN: nvq++ --enable-mlir --target remote-mqpu %s -o %t  && %t | FileCheck %s
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ --enable-mlir --target remote-mqpu -fkernel-exec-kind=2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <iostream>

// This test allocates 524288 doubles (4MB) on the stack, which requires
// an 8MB+ thread stack size (Linux default) to avoid stack overflow.
__qpu__ void test_large_double_constant_array() {
  std::vector<double> vec(1ULL << 19);  // 524288 elements = 4MB
  vec[0] = M_SQRT1_2 / vec.size();
  vec[1] = M_SQRT1_2 / vec.size();
  for (std::size_t i = 2; i < vec.size(); i++) {
    vec[i] = 0;
  }
  cudaq::qvector v(vec);
}

void printCounts(cudaq::sample_result &result) {
  std::vector<std::string> values{};
  for (auto &&[bits, counts] : result) {
    values.push_back(bits);
  }

  std::sort(values.begin(), values.end());
  for (auto &&bits : values) {
    std::cout << bits << '\n';
  }
}

int main() {
  auto counts = cudaq::sample(test_large_double_constant_array);
  std::cout << "Large array test\n";
  printCounts(counts);

  // CHECK-LABEL: Large array test
  // CHECK: 0000000000000000000
  // CHECK: 1000000000000000000
}

