/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/state.h"
#include "cudaq/builder/kernels.h"
#include <iostream>

namespace cudaq {
namespace details {
std::vector<std::string> grayCode(std::size_t);
}
} // namespace cudaq

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
    auto test = cudaq::details::getControlIndices(2);
    std::vector<std::size_t> expected{0, 1, 0, 1};
    EXPECT_EQ(test.size(), expected.size());
    EXPECT_EQ(test, expected);
  }
  {
    std::vector<std::size_t> expected{0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
                                      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
                                      0, 3, 0, 1, 0, 2, 0, 1, 0, 4};

    auto test = cudaq::details::getControlIndices(5);
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
    }
  }

  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 1);
    std::vector<double> expected{0.0, 3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

CUDAQ_TEST(KernelsTester, checkGetAlphaZ) {
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 3);
    std::vector<double> expected{1.17809725};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 2);
    std::vector<double> expected{0., 0.78539816};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }

  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 1);
    std::vector<double> expected{0., 0., 1.57079633, -3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(KernelsTester, checkFromState) {
  {
    std::vector<std::complex<double>> state{.70710678, 0., 0., 0.70710678};
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);

    cudaq::from_state(kernel, qubits, state);

    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel);
    counts.dump();
  }

  {
    std::vector<std::complex<double>> state{0., .292786, .956178, 0.};
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);
    cudaq::from_state(kernel, qubits, state);
    std::cout << kernel << "\n";
    using namespace cudaq::spin;
    auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
             .21829 * z(0) - 6.125 * z(1);
    auto energy = cudaq::observe(kernel, H).exp_val_z();
    EXPECT_NEAR(-1.748, energy, 1e-3);

    auto ss = cudaq::get_state(kernel);
    for (std::size_t i = 0; i < 4; i++)
      EXPECT_NEAR(ss[i].real(), state[i].real(), 1e-3);
  }

  {
    std::vector<std::complex<double>> state{.70710678, 0., 0., 0.70710678};

    // Return a kernel from the given state, this
    // comes back as a unique_ptr.
    auto kernel = cudaq::from_state(state);
    std::cout << *kernel << "\n";
    auto counts = cudaq::sample(*kernel);
    counts.dump();
  }
}

#endif