/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include <cmath>
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <gtest/gtest.h>
#include <random>

class TestEnvironment : public ::testing::Environment {
protected:
  void SetUp() override { cudaq::mpi::initialize(); }
  void TearDown() override { cudaq::mpi::finalize(); }
};

::testing::Environment *const foo_env =
    AddGlobalTestEnvironment(new TestEnvironment);

TEST(MPIObserveTester, checkSimple) {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  auto result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
  if (cudaq::mpi::rank() == 0) {
    EXPECT_NEAR(result.expectation(), -1.7487, 1e-3);
    printf("Get energy directly as double %lf\n", result.expectation());
  }

  // Every rank has the expectation value of every term, as with threads.
  auto threadResult = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
  for (const auto &term : h)
    EXPECT_NEAR(result.expectation(term), threadResult.expectation(term), 1e-6);

  // The ansatz prepares cos(t/2)|10> + sin(t/2)|01> with t = 0.59.
  const double t = 0.59;
  EXPECT_NEAR(result.expectation(cudaq::spin_op::x(0) * cudaq::spin_op::x(1)),
              std::sin(t), 1e-5);
  EXPECT_NEAR(result.expectation(cudaq::spin_op::y(0) * cudaq::spin_op::y(1)),
              std::sin(t), 1e-5);
  EXPECT_NEAR(result.expectation(cudaq::spin_op::z(0)), -std::cos(t), 1e-5);
  EXPECT_NEAR(result.expectation(cudaq::spin_op::z(1)), std::cos(t), 1e-5);
}
