/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"

namespace {
auto mappedPartialKernel() {
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(3);
  kernel.x(qubits[0]);
  kernel.x(qubits[2]);
  kernel.z<cudaq::ctrl>(qubits[0], qubits[2]);
  kernel.mz(qubits[1]);
  kernel.mz(qubits[0]);
  return kernel;
}

void expectMappedPartialCounts(const cudaq::sample_result &result) {
  EXPECT_EQ(result.get_total_shots(), 5);
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.count("10"), 5);
}

} // namespace

CUDAQ_TEST(LQCloud, SampleAndSampleAsync) {
  auto kernel = mappedPartialKernel();
  expectMappedPartialCounts(cudaq::sample(5, kernel));
  expectMappedPartialCounts(cudaq::sample_async(5, 0, kernel).get());
}
