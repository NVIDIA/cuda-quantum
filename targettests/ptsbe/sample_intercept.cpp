/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// PTSBE sample intercept POC test. Verifies that sampleWithPTSBE can capture
// trace from kernel and construct PTSBatch. Full execution is not implemented.

// RUN: nvq++ --enable-mlir %s -o %t && %t

#include <cudaq.h>
#include <cudaq/ptsbe/PTSBESample.h>

struct bellKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    mz(q);
  }
};

int main() {
  bellKernel kernel;

  // capturePTSBatch should work for valid kernels
  auto batch = cudaq::ptsbe::capturePTSBatch(kernel);

  // Verify batch has captured trace
  std::size_t count = 0;
  for (const auto &inst : batch.kernel_trace) {
    (void)inst;
    ++count;
  }

  if (count != 2) {
    printf("FAIL: Expected 2 instructions, got %zu\n", count);
    return 1;
  }

  if (batch.measure_qubits.size() != 2) {
    printf("FAIL: Expected 2 measure qubits, got %zu\n",
           batch.measure_qubits.size());
    return 1;
  }

  printf("PASS: PTSBatch captured with %zu instructions and %zu measure qubits\n",
         count, batch.measure_qubits.size());
  return 0;
}
