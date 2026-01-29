/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// PTSBE does not support mid-circuit measurements. This test verifies that
// sampleWithPTSBE rejects kernels with conditional feedback on measure results.
//
// Pending implementation of MCM detection.

// RUN: nvq++ --enable-mlir %s -o %t && %t

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

  // sampleWithPTSBE should throw for MCM circuits with a specific error
  try {
    cudaq::ptsbe::sampleWithPTSBE(kernel, 1000);
    printf("FAIL: Expected exception for MCM kernel\n");
    return 1;
  } catch (const std::runtime_error &e) {
    // Verify the error is from MCM detection, not dispatchPTSBE
    if (std::strstr(e.what(), "mid-circuit measurements") != nullptr) {
      printf("PASS: MCM kernel rejected with correct error: %s\n", e.what());
      return 0;
    } else {
      printf("FAIL: Wrong error (expected MCM rejection): %s\n", e.what());
      return 1;
    }
  }
}
