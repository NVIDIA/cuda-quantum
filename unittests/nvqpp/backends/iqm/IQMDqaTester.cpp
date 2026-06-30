/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "CUDAQTestUtils.h"
#include "cudaq/algorithm.h"

CUDAQ_TEST(IQMDqaTester, dynamicQuantumArchitectureFile) {
  auto kernel3 = cudaq::make_kernel();
  auto qubit3 = kernel3.qalloc(2);
  kernel3.h(qubit3[0]);
  kernel3.mz(qubit3[0]);
  kernel3.mz(qubit3[1]);

  auto counts = cudaq::sample(kernel3);

  EXPECT_GE(counts.size(), 2);
  EXPECT_LE(counts.size(), 4);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
