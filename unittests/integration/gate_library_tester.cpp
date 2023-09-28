/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/state.h"
#include "cudaq/qis/library/givens_rotation.h"
#include <random>
using namespace cudaq;

CUDAQ_TEST(GateLibraryTester, checkGivensRotation) {
  std::random_device rd;
  std::default_random_engine re(rd());
  std::uniform_real_distribution<double> dist(-M_PI, M_PI);
  const double angle = dist(re);
  std::cout << "Angle = " << angle << "\n";
  auto test_01 = [](double theta) __qpu__ {
    cudaq::qreg<2> q;
    x(q[0]);
    cudaq::givens_rotation{}(theta, q[0], q[1]);
  };
  auto test_10 = [](double theta) __qpu__ {
    cudaq::qreg<2> q;
    x(q[1]);
    cudaq::givens_rotation{}(theta, q[0], q[1]);
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
  EXPECT_NEAR(std::abs(ss_01[1] + s), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(ss_01[2] - c), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(ss_10[1] - c), 0.0, 1e-9);
  EXPECT_NEAR(std::abs(ss_10[2] - s), 0.0, 1e-9);
}