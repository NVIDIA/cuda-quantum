/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --enable-mlir --target quantinuum --emulate %s -o %t && \
// RUN: rm -f %t.log && \
// RUN: CUDAQ_PIPELINE_LOG=%t.log %t && \
// RUN: FileCheck %s --input-file=%t.log
// clang-format on

#include <cudaq.h>

int main() {
  auto sampleKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
  };
  auto observeKernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
  };

  cudaq::sample(sampleKernel);
  cudaq::observe(observeKernel, cudaq::spin_op::from_word("Z"));
  // CHECK: "type":"configured"
  // CHECK: "label":"
  // CHECK: "pipeline":"
  // CHECK: "type":"executed"
  // CHECK: "label":"
  // CHECK: "passes":[{"pass":"
  return 0;
}
