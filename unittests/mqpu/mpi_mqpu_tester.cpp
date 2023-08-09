/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
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
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
  if (cudaq::mpi::rank() == 0) {
    EXPECT_NEAR(result, -1.7487, 1e-3);
    printf("Get energy directly as double %lf\n", result);
  }
}
