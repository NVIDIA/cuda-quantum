/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>

#ifndef CUDAQ_BACKEND_DM

#ifndef CUDAQ_BACKEND_STIM
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
  std::vector<double> expected{-0.436289, 1.299263, 3.534137, 6.026147,
                              8.505245, 10.702783,  12.380624,  13.356947,
                              13.525953,  12.869326,  11.458224, 9.445560,
                              7.049439, 4.529516, 2.158865,  0.194383,
                              -1.151048,  -1.731631,  -1.484449, -0.436289};
  for (auto &r : results) {
    EXPECT_NEAR(expected[i], r.second.get(), 1e-3);
    i++;
  }
}
#endif

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

#ifndef CUDAQ_BACKEND_STIM
CUDAQ_TEST(AsyncTester, checkGetStateAsync) {
  struct ghz {
    auto operator()(int NQubits) __qpu__ {
      // int N = 5;
      cudaq::qvector q(NQubits);
      h(q[0]);
      for (int i = 0; i < NQubits - 1; i++) {
        x<cudaq::ctrl>(q[i], q[i + 1]);
      }
    }
  };

  auto cc0 = cudaq::get_state_async(0, ghz{}, 5);
  auto cc1 = cudaq::get_state_async(0, ghz{}, 5);
  auto cc2 = cudaq::get_state_async(0, ghz{}, 5);
  // run the the zeroth one
  auto cc3 = cudaq::get_state_async(ghz{}, 5);
  auto cc0State = cc0.get();
  auto cc1State = cc1.get();
  auto cc2State = cc2.get();
  auto cc3State = cc3.get();

  if (cc0State.get_precision() == cudaq::SimulationState::precision::fp32) {
    std::vector<std::complex<float>> expectedVec(1 << 5, 0.0);
    expectedVec[0] = M_SQRT1_2;
    expectedVec[expectedVec.size() - 1] = M_SQRT1_2;

    auto expectedState = cudaq::state::from_data(expectedVec);

    EXPECT_NEAR(cc0State.overlap(expectedState).real(), 1.0, 1e-3);
    EXPECT_NEAR(cc1State.overlap(expectedState).real(), 1.0, 1e-3);
    EXPECT_NEAR(cc2State.overlap(expectedState).real(), 1.0, 1e-3);
    EXPECT_NEAR(cc3State.overlap(expectedState).real(), 1.0, 1e-3);
  }
}
#endif
#endif
