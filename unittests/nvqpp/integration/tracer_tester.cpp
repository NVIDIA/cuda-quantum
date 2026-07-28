/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithms/resource_estimation.h>
#include <random>
#include <stdio.h>

CUDAQ_TEST(TracerTester, checkBell) {

  auto bell = []() __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
  };

  auto resources = cudaq::estimate_resources(bell);
  resources.dump();

  // Count how many hadamards we have
  EXPECT_EQ(1, resources.count("h"));

  // Count how many ctrl-x gates we have with one control
  EXPECT_EQ(1, resources.count_controls("x", /*controls*/ 1));

  // Count how many rx gates we have
  EXPECT_EQ(0, resources.count("rx"));
}

CUDAQ_TEST(TracerTester, checkGHZ) {

  auto ghz = [](int i) __qpu__ {
    cudaq::qvector q(i);
    h(q[0]);
    for (int j = 0; j < i - 1; j++)
      x<cudaq::ctrl>(q[j], q[j + 1]);
  };

  auto resources = cudaq::estimate_resources(ghz, 10);
  resources.dump();

  // How many hadamards?
  EXPECT_EQ(1, resources.count("h"));
  // How many ctrl-x operations with 1 ctrl qubit
  EXPECT_EQ(9, resources.count_controls("x", /*nControls*/ 1));
  // How many x operations, any number of controls
  EXPECT_EQ(9, resources.count("x"));
}

CUDAQ_TEST(TracerTester, checkLargeTrace) {

  auto largeTrace = [](int numQubits, int numLayers, std::vector<int> cnotPairs)
                        __qpu__ {
                          cudaq::qvector q(numQubits);

                          for (int layer = 0; layer < numLayers; layer++) {
                            // each layer should be composed of a set of random
                            // single qubit gates on every qubit, followed by
                            // a layer of random cnots
                            for (int i = 0; i < numQubits; i++) {
                              int choice = cnotPairs[layer * numQubits + i] % 6;
                              if (choice == 0)
                                h(q[i]);
                              else if (choice == 1)
                                x(q[i]);
                              else if (choice == 2)
                                y(q[i]);
                              else if (choice == 3)
                                z(q[i]);
                              else if (choice == 4)
                                s(q[i]);
                              else
                                t(q[i]);
                            }

                            for (int i = 0; i < numQubits; i += 2)
                              x<cudaq::ctrl>(q[i], q[i + 1]);
                          }
                        };

  int numQubits = 1000;
  int numLayers = 1000;

  std::vector<int> cnots(numQubits * numLayers);
  std::iota(cnots.begin(), cnots.end(), 0);
  std::shuffle(cnots.begin(), cnots.end(),
               std::mt19937{std::random_device{}()});
  auto resources =
      cudaq::estimate_resources(largeTrace, numQubits, numLayers, cnots);
  auto totalOps = resources.count();
  EXPECT_EQ(totalOps, numLayers * numQubits * 1.5);
}
