/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "cudaq/algorithm.h"
#include "math.h"
#include <fstream>
#include <gtest/gtest.h>
#include <stdlib.h>

std::string mockPort = "62448";
std::string backendStringTemplate = "quantum_machines;url;http://localhost:{}";

CUDAQ_TEST(QuantumMachinesTester, minimal3Hadamard) {
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(3);
  kernel.h(qubit[0]);
  kernel.h(qubit[1]);
  kernel.h(qubit[2]);

  auto counts = cudaq::sample(1000, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

CUDAQ_TEST(QuantumMachinesTester, resetAndH) {
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(3);
  for (int i = 0; i < 3; i++) {
    kernel.reset(qubit[0]);
  }
  for (int i = 0; i < 3; i++) {
    kernel.h(qubit[1]);
  }

  auto counts = cudaq::sample(1000, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

CUDAQ_TEST(QuantumMachinesTester, gates) {
  auto backendString =
      fmt::format(fmt::runtime(backendStringTemplate), mockPort);

  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();

  int qubit_count = 5;
  auto qvector = kernel.qalloc(qubit_count);
  for (int i = 0; i < qubit_count; i++) {
    kernel.reset(qvector[i]);
  }
  kernel.t(qvector[0]);
  kernel.s(qvector[1]);
  kernel.r1(1.1853982, qvector[2]);
  kernel.x(qvector[3]);
  kernel.y(qvector[4]);

  auto counts = cudaq::sample(1001, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 8);
}

int main(int argc, char **argv) {
  setenv("QUANTUM_MACHINES_API_KEY", "00000000000000000000000000000000", 0);
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
