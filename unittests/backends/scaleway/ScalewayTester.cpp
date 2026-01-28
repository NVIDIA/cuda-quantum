
/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gmock/gmock-matchers.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <regex>

std::string backendString = "scaleway;emulate;false;url;"
                            "http://localhost:62443";

CUDAQ_TEST(ScalewayTester, executeOneMeasuredQubitProgram) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend(backendString);

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.y(qubit[0]);
  kernel.z(qubit[0]);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(kernel);
  EXPECT_EQ(counts.size(), 2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleMock(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
