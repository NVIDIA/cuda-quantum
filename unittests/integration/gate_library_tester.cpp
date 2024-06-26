/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/kernels/fermionic_swap.h"
#include "cudaq/kernels/givens_rotation.h"
#include <random>
using namespace cudaq;

#ifndef CUDAQ_BACKEND_DM
namespace {
// These tests are meant to validate the correctness of custom kernels.
// Hence, reduce the test load on tensornet backends (slow for these small
// circuits). The circuit correctness is validated more thoroughly by
// state-vector based backends.
#ifdef CUDAQ_BACKEND_TENSORNET
constexpr size_t NUM_ANGLES = 10;
#else
constexpr size_t NUM_ANGLES = 100;
#endif
} // namespace

CUDAQ_TEST(GateLibraryTester, checkGivensRotation) {
  for (const auto &angle : cudaq::linspace(-M_PI, M_PI, NUM_ANGLES)) {
    auto test_01 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      x(q[0]);
      cudaq::givens_rotation(theta, q[0], q[1]);
    };
    auto test_10 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      x(q[1]);
      cudaq::givens_rotation(theta, q[0], q[1]);
    };
    // Matrix
    //    [[1, 0, 0, 0],
    //     [0, c, -s, 0],
    //     [0, s, c, 0],
    //     [0, 0, 0, 1]]
    // where c = cos(theta); s = sin(theta)
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    auto ss_01 = cudaq::get_state(test_01, angle);
    auto ss_10 = cudaq::get_state(test_10, angle);
    EXPECT_NEAR(std::abs(ss_01[1] + s), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(ss_01[2] - c), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(ss_10[1] - c), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(ss_10[2] - s), 0.0, 1e-6);
  }
}

CUDAQ_TEST(GateLibraryTester, checkGivensRotationKernelBuilder) {
  for (const auto &angle : cudaq::linspace(-M_PI, M_PI, NUM_ANGLES)) {
    // Matrix
    //    [[1, 0, 0, 0],
    //     [0, c, -s, 0],
    //     [0, s, c, 0],
    //     [0, 0, 0, 1]]
    // where c = cos(theta); s = sin(theta)
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    {
      auto [test_01, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_01.qalloc(2);
      test_01.x(q[0]);
      cudaq::builder::givens_rotation(test_01, theta, q[0], q[1]);
      auto ss_01 = cudaq::get_state(test_01, angle);
      EXPECT_NEAR(std::abs(ss_01[1] + s), 0.0, 1e-6);
      EXPECT_NEAR(std::abs(ss_01[2] - c), 0.0, 1e-6);
    }
    {
      auto [test_10, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_10.qalloc(2);
      test_10.x(q[1]);
      cudaq::builder::givens_rotation(test_10, theta, q[0], q[1]);
      auto ss_10 = cudaq::get_state(test_10, angle);
      EXPECT_NEAR(std::abs(ss_10[1] - c), 0.0, 1e-6);
      EXPECT_NEAR(std::abs(ss_10[2] - s), 0.0, 1e-6);
    }
  }
}

CUDAQ_TEST(GateLibraryTester, checkControlledGivensRotation) {
  for (const auto &angle : cudaq::linspace(-M_PI, M_PI, NUM_ANGLES)) {
    // Same check, with 2 control qubits
    auto test_01_on = [](double theta) __qpu__ {
      cudaq::qarray<4> q;
      x(q[2]);
      x(q[0]);
      x(q[1]);
      cudaq::control(cudaq::givens_rotation, {q[0], q[1]}, theta, q[2], q[3]);
    };

    auto test_01_off = [](double theta) __qpu__ {
      cudaq::qarray<4> q;
      x(q[2]);
      cudaq::control(cudaq::givens_rotation, {q[0], q[1]}, theta, q[2], q[3]);
      x(q[2]);
    };

    const double c = std::cos(angle);
    const double s = std::sin(angle);
    auto ss_01_on = cudaq::get_state(test_01_on, angle);
    EXPECT_NEAR(std::abs(ss_01_on[13] + s), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(ss_01_on[14] - c), 0.0, 1e-6);
    auto ss_01_off = cudaq::get_state(test_01_off, angle);
    EXPECT_NEAR(std::abs(ss_01_off[0]), 1.0, 1e-6);
  }
}

CUDAQ_TEST(GateLibraryTester, checkFermionicSwap) {
  for (const auto &angle : cudaq::linspace(-M_PI, M_PI, NUM_ANGLES)) {
    auto test_00 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      cudaq::fermionic_swap(theta, q[0], q[1]);
    };
    auto test_01 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      x(q[0]);
      cudaq::fermionic_swap(theta, q[0], q[1]);
    };
    auto test_10 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      x(q[1]);
      cudaq::fermionic_swap(theta, q[0], q[1]);
    };
    auto test_11 = [](double theta) __qpu__ {
      cudaq::qarray<2> q;
      x(q);
      cudaq::fermionic_swap(theta, q[0], q[1]);
    };

    // FermionicSWAP truth table
    // |00⟩ ↦ |00⟩
    // |01⟩ ↦ e^(iϕ/2)cos(ϕ/2)|01⟩ − ie^(iϕ/2)sin(ϕ/2)|10⟩
    // |10⟩ ↦ −i^(eiϕ/2)sin(ϕ/2)|01⟩ + e^(iϕ/2)cos(ϕ/2)|10⟩
    // |11⟩ ↦ e^(iϕ)|11⟩,

    const double c = std::cos(angle / 2.0);
    const double s = std::sin(angle / 2.0);
    constexpr std::complex<double> I{0.0, 1.0};
    {
      auto ss_00 = cudaq::get_state(test_00, angle);
      EXPECT_NEAR(std::norm(ss_00[0] - 1.0), 0.0, 1e-6);
    }
    {
      auto ss_01 = cudaq::get_state(test_01, angle);
      EXPECT_NEAR(std::norm(ss_01[1] - (-I * std::exp(I * angle / 2.0) * s)),
                  0.0, 1e-6);
      EXPECT_NEAR(std::norm(ss_01[2] - (std::exp(I * angle / 2.0) * c)), 0.0,
                  1e-6);
    }
    {
      auto ss_10 = cudaq::get_state(test_10, angle);
      EXPECT_NEAR(std::norm(ss_10[1] - (std::exp(I * angle / 2.0) * c)), 0.0,
                  1e-6);
      EXPECT_NEAR(std::norm(ss_10[2] - ((-I * std::exp(I * angle / 2.0) * s))),
                  0.0, 1e-6);
    }
    {
      // |11⟩ ↦ e^(iϕ)|11⟩
      // (|11> has an extra global phase)
      auto ss_11 = cudaq::get_state(test_11, angle);
      EXPECT_NEAR(std::norm(ss_11[3] - std::exp(I * angle)), 0.0, 1e-6);
    }
  }
}

CUDAQ_TEST(GateLibraryTester, checkFermionicSwapKernelBuilder) {
  for (const auto &angle : cudaq::linspace(-M_PI, M_PI, NUM_ANGLES)) {
    const double c = std::cos(angle / 2.0);
    const double s = std::sin(angle / 2.0);
    constexpr std::complex<double> I{0.0, 1.0};
    {
      auto [test_00, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_00.qalloc(2);
      cudaq::builder::fermionic_swap(test_00, theta, q[0], q[1]);
      auto ss_00 = cudaq::get_state(test_00, angle);
      EXPECT_NEAR(std::norm(ss_00[0] - 1.0), 0.0, 1e-6);
    }
    {
      auto [test_01, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_01.qalloc(2);
      test_01.x(q[0]);
      cudaq::builder::fermionic_swap(test_01, theta, q[0], q[1]);
      auto ss_01 = cudaq::get_state(test_01, angle);
      EXPECT_NEAR(std::norm(ss_01[1] - (-I * std::exp(I * angle / 2.0) * s)),
                  0.0, 1e-6);
      EXPECT_NEAR(std::norm(ss_01[2] - (std::exp(I * angle / 2.0) * c)), 0.0,
                  1e-6);
    }
    {
      auto [test_10, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_10.qalloc(2);
      test_10.x(q[1]);
      cudaq::builder::fermionic_swap(test_10, theta, q[0], q[1]);
      auto ss_10 = cudaq::get_state(test_10, angle);
      EXPECT_NEAR(std::norm(ss_10[1] - (std::exp(I * angle / 2.0) * c)), 0.0,
                  1e-6);
      EXPECT_NEAR(std::norm(ss_10[2] - ((-I * std::exp(I * angle / 2.0) * s))),
                  0.0, 1e-6);
    }
    {
      auto [test_11, theta] = cudaq::make_kernel<double>();
      // Allocate some qubits
      auto q = test_11.qalloc(2);
      test_11.x(q);
      cudaq::builder::fermionic_swap(test_11, theta, q[0], q[1]);
      auto ss_11 = cudaq::get_state(test_11, angle);
      EXPECT_NEAR(std::norm(ss_11[3] - std::exp(I * angle)), 0.0, 1e-6);
    }
  }
}
#endif
