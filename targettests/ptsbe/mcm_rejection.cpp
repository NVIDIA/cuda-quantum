/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// PTSBE does not support dynamic circuits. This test verifies that
// sampleWithPTSBE rejects kernels with conditional feedback on measure results.

// RUN: nvq++ --enable-mlir %s -o %t && %t | FileCheck %s

// CHECK: PASS: Dynamic circuit rejected

#include <cudaq.h>
#include <cudaq/ptsbe/PTSBESample.h>
#include <cstring>

struct mcmKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    auto result = mz(q[0]);
    if (result) {
      x(q[1]);
    }
    mz(q[1]);
  }
};

int main() {
  mcmKernel kernel;

  // sampleWithPTSBE should throw for dynamic circuits with a specific error
  try {
    cudaq::ptsbe::sampleWithPTSBE(kernel, 1000);
    printf("FAIL: Expected exception for dynamic circuit kernel\n");
    return 1;
  } catch (const std::runtime_error &e) {
    // Verify the error is from dynamic circuit detection, not dispatchPTSBE
    if (std::strstr(e.what(), "dynamic circuits") != nullptr) {
      printf("PASS: Dynamic circuit rejected with correct error: %s\n", e.what());
      return 0;
    } else {
      printf("FAIL: Wrong error (expected dynamic circuit rejection): %s\n", e.what());
      return 1;
    }
  }
}
