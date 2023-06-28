/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "QPPObserveBackend.cpp"
#include "cudaq/algorithm.h"
#include <gtest/gtest.h>

CUDAQ_TEST(QPPBackendTester, checkBackendObserve) {

  cudaq::QppObserveTester qpp;
  auto q0 = qpp.allocateQubit();
  auto q1 = qpp.allocateQubit();

  qpp.x(q0);
  qpp.ry(.59, q1);
  qpp.x({q1}, q0);

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto expVal = qpp.observe(h);
  EXPECT_NEAR(expVal.expectationValue.value(), -1.74, 1e-2);

  struct ansatzTest {
    auto operator()(double theta) __qpu__ {
      // Programmer would just write this...
      cudaq::qvector q(2);
      x(q[0]);
      ry(theta, q[1]);
      x<cudaq::ctrl>(q[1], q[0]);
    }
  };

  double energy = cudaq::observe(ansatzTest{}, h, .59);
  EXPECT_NEAR(energy, -1.74, 1e-2);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
