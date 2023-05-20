/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithms/resource_estimation.h>
#include <stdio.h>

CUDAQ_TEST(TracerTester, checkBell) {

  auto bell = []() __qpu__ {
    cudaq::qreg q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  auto resources = cudaq::estimate_resources(bell);
  resources.dump();

  // Count how many hadamards we have on qubit 0
  EXPECT_EQ(1, resources.count("h", /*qubit*/ 0));

  // Count how many hadamards we have on any qubit
  EXPECT_EQ(1, resources.count("h"));

  // Count how many x gates we have on qubit 1
  EXPECT_EQ(1, resources.count_controls("x", /*nControls*/ 1));

  // Count how many ctrl-x gates we have between 0 and 1
  EXPECT_EQ(1, resources.count("x", /*controls*/ {0}, /*qubit*/ 1));

  // Count how many rx gates we have
  EXPECT_EQ(0, resources.count("rx"));
}

CUDAQ_TEST(TracerTester, checkGHZ) {

  auto ghz = [](int i) __qpu__ {
    cudaq::qreg q(i);
    h(q[0]);
    for (int j = 0; j < i - 1; j++)
      x<cudaq::ctrl>(q[j], q[j + 1]);
  };

  auto resources = cudaq::estimate_resources(ghz, 10);
  resources.dump();

  // How many hadamards on qubit 0
  EXPECT_EQ(1, resources.count("h", /*qubit*/ 0));
  // How many hadamards?
  EXPECT_EQ(1, resources.count("h"));
  // How many ctrl-x operations with 1 ctrl qubit
  EXPECT_EQ(9, resources.count_controls("x", /*nControls*/ 1));
  // How many x operations, any number of controls
  EXPECT_EQ(9, resources.count("x"));
}
