/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/builder/kernels.h"
#include <iostream>

CUDAQ_TEST(KernelsTester, checkGrayCode) {
  {
    auto test = cudaq::details::grayCode(2);
    std::vector<std::string> expected{"00", "01", "11", "10"};
    EXPECT_EQ(test.size(), expected.size());
    for (auto &t : test) {
      EXPECT_TRUE(std::find(expected.begin(), expected.end(), t) !=
                  expected.end());
    }
  }
  {
    std::vector<std::string> expected{
        "00000", "00001", "00011", "00010", "00110", "00111", "00101", "00100",
        "01100", "01101", "01111", "01110", "01010", "01011", "01001", "01000",
        "11000", "11001", "11011", "11010", "11110", "11111", "11101", "11100",
        "10100", "10101", "10111", "10110", "10010", "10011", "10001", "10000"};

    auto test = cudaq::details::grayCode(5);
    EXPECT_EQ(test.size(), expected.size());
    for (auto &t : test) {
      EXPECT_TRUE(std::find(expected.begin(), expected.end(), t) !=
                  expected.end());
    }
  }
}

CUDAQ_TEST(KernelsTester, checkGenCtrlIndices) {
  {
    auto test = cudaq::details::getControlIndicesFromGrayCode(2);
    std::vector<std::size_t> expected{0, 1, 0, 1};
    EXPECT_EQ(test.size(), expected.size());
    EXPECT_EQ(test, expected);
  }
  {
    std::vector<std::size_t> expected{0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
                                      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
                                      0, 3, 0, 1, 0, 2, 0, 1, 0, 4};

    auto test = cudaq::details::getControlIndicesFromGrayCode(5);
    EXPECT_EQ(test.size(), expected.size());
    EXPECT_EQ(test, expected);
  }
}

CUDAQ_TEST(KernelsTester, checkGetAlphaY) {
  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 2);
    std::vector<double> expected{1.57079633};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
      std::cout << t << "\n";
    }
  }

  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 1);
    std::vector<double> expected{0.0, 3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
      std::cout << t << "\n";
    }
  }
}

CUDAQ_TEST(KernelsTester, checkFromState) {
  {
    std::vector<std::complex<double>> state{.70710678, 0., 0., 0.70710678};
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);

    cudaq::from_state(kernel, qubits, state, cudaq::range(2));

    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel);
    counts.dump();
  }
}