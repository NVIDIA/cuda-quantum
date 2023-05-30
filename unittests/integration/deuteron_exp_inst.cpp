/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

struct ansatz {
  auto operator()(double theta) __qpu__ {
    using namespace cudaq::spin;
    cudaq::qreg q(2);
    x(q[0]);
    exp(q, theta, x(0) * y(1) - y(0) * x(1));
  }
};

CUDAQ_TEST(D2ExpObserveTester, checkSimple) {
  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  h.dump();

  double energy = cudaq::observe(ansatz{}, h, .297209);
  printf("Energy is %lf\n", energy);
  EXPECT_NEAR(energy, -1.7487, 1e-3);
}
