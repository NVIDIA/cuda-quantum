/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

struct ansatz2 {
  auto operator()(double theta) __qpu__ {
    cudaq::qreg q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

CUDAQ_TEST(D2VariationalTester, checkSimple) {

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  h.dump();

  double energy = cudaq::observe(ansatz2{}, h, .59);
  printf("Energy is %lf\n", energy);
  EXPECT_NEAR(energy, -1.7487, 1e-3);
}

CUDAQ_TEST(D2VariationalTester, checkBroadcast) {

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto params = cudaq::linspace(-M_PI, M_PI, 50);

  auto ansatz = [](double theta, int size) __qpu__ {
    cudaq::qreg q(size);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  auto results = cudaq::observe_n(
      ansatz, h, cudaq::make_argset(params, std::vector(params.size(), 2)));

  std::vector<double> expected{
      12.250290, 12.746370, 13.130148, 13.395321, 13.537537, 13.554460,
      13.445811, 13.213375, 12.860969, 12.394379, 11.821267, 11.151042,
      10.394710, 9.564690,  8.674611,  7.739088,  6.773482,  5.793648,
      4.815676,  3.855623,  2.929254,  2.051779,  1.237607,  0.500106,
      -0.148614, -0.697900, -1.138735, -1.463879, -1.667993, -1.747726,
      -1.701768, -1.530875, -1.237852, -0.827511, -0.306589, 0.316359,
      1.031106,  1.825915,  2.687735,  3.602415,  4.554937,  5.529659,
      6.510578,  7.481585,  8.426738,  9.330517,  10.178082, 10.955516,
      11.650053, 12.250290};

  for (std::size_t counter = 0; auto &el : expected)
    EXPECT_NEAR(results[counter++].exp_val_z(), el, 1e-3);

  // Expect that providing the wrong number of args in the vector will
  // throw an exception.
  EXPECT_ANY_THROW({
    auto results = cudaq::observe_n(
        ansatz, h,
        cudaq::make_argset(params, std::vector(params.size() + 1, 2)));
  });
}
