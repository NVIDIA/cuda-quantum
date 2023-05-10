/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(AsyncTester, checkObserveAsync) {

  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  h.dump();

  auto params = cudaq::linspace(-M_PI, M_PI, 20);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  };

  std::vector<std::pair<double, cudaq::async_observe_result>> results;
  for (auto &param : params) {
    results.emplace_back(
        std::make_pair(param, cudaq::observe_async(ansatz, h, param)));
  }

  auto i = 0;
  std::vector<double> expected{12.250290, 13.298450, 13.545631, 12.965049,
                               11.619617, 9.655134,  7.284483,  4.764561,
                               2.368439,  0.355775,  -1.055327, -1.711953,
                               -1.542948, -0.566625, 1.111216,  3.308754,
                               5.787853,  8.279863,  10.514736, 12.250290};
  for (auto &r : results) {
    EXPECT_NEAR(expected[i], r.second.get(), 1e-3);
    i++;
  }
}

CUDAQ_TEST(AsyncTester, checkSampleAsync) {
  struct ghz {
    auto operator()(int NQubits) __qpu__ {
      // int N = 5;
      cudaq::qvector q(NQubits);
      h(q[0]);
      for (int i = 0; i < NQubits - 1; i++) {
        x<cudaq::ctrl>(q[i], q[i + 1]);
      }
      mz(q);
    }
  };

  auto cc0 = cudaq::sample_async(0, ghz{}, 5);
  auto cc1 = cudaq::sample_async(0, ghz{}, 5);
  auto cc2 = cudaq::sample_async(0, ghz{}, 5);
  // run the the zeroth one
  auto cc3 = cudaq::sample_async(ghz{}, 5);

  cc0.get().dump();
  cc1.get().dump();
  cc2.get().dump();
  cc3.get().dump();
}
#endif
