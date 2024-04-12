/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include <numeric>

using namespace cudaq;

CUDAQ_TEST(GetStateTester, checkSimple) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    cx(q, r);
  };

  auto state = cudaq::get_state(kernel);
  state.dump();
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(0.5, state(0, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(0, 3).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 3).real(), 1e-3);

  EXPECT_NEAR(0.5, state.amplitude({0, 0}).real(), 1e-3);
  EXPECT_NEAR(0.0, state.amplitude({1, 0}).real(), 1e-3);
  EXPECT_NEAR(0.0, state.amplitude({0, 1}).real(), 1e-3);
  EXPECT_NEAR(0.5, state.amplitude({1, 1}).real(), 1e-3);
#else
  EXPECT_NEAR(1. / std::sqrt(2.), state[0].real(), 1e-3);
  EXPECT_NEAR(0., state[1].real(), 1e-3);
  EXPECT_NEAR(0., state[2].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state[3].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state.amplitude({0, 0}).real(), 1e-3);
  EXPECT_NEAR(0.0, state.amplitude({1, 0}).real(), 1e-3);
  EXPECT_NEAR(0.0, state.amplitude({0, 1}).real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state.amplitude({1, 1}).real(), 1e-3);
#endif

  EXPECT_NEAR(state.overlap(state).real(), 1.0, 1e-3);

  auto kernelNoop = []() __qpu__ {
    cudaq::qubit q, r;
    x(q);
    x(q);
  };

  // Check <00|Bell> = 1/sqrt(2)
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelNoop)).real(),
              1. / std::sqrt(2.), 1e-3);

  auto kernelX = []() __qpu__ {
    cudaq::qubit q, r;
    x(q);
    x(r);
  };

  // Check <11|Bell> = 1/sqrt(2)
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelX)).real(),
              1. / std::sqrt(2.), 1e-3);

  // Check endianess of basis state
  auto kernel1 = []() __qpu__ {
    cudaq::qvector qvec(5);
    x(qvec[1]);
    x(qvec[2]);
  };

  auto state1 = cudaq::get_state(kernel1);
  EXPECT_NEAR(1.0, state1.amplitude({0, 1, 1, 0, 0}).real(), 1e-3);

#ifndef CUDAQ_BACKEND_TENSORNET
  // Demonstrate a useful use-case for get_state,
  // specifically, let's approximate another 2-qubit state with a
  // general so4 rotation. Here we'll see if we can find rotational
  // parameters that create a circuit producing the bell state.
  auto so4 = [](std::vector<double> parameters) __qpu__ {
    cudaq::qubit q, r;
    ry(parameters[0], q);
    ry(parameters[1], r);

    z<cudaq::ctrl>(q, r);

    ry(parameters[2], q);
    ry(parameters[3], r);

    z<cudaq::ctrl>(q, r);

    ry(parameters[4], q);
    ry(parameters[5], r);

    z<cudaq::ctrl>(q, r);
  };

  cudaq::optimizers::cobyla optimizer;
  optimizer.max_eval = 100;
  auto [opt_val, params] = optimizer.optimize(6, [&](std::vector<double> x) {
    auto testState = cudaq::get_state(so4, x);
    return 1.0 - state.overlap(testState).real();
  });

  EXPECT_NEAR(opt_val, 0.0, 1e-3);
#endif
}

#ifdef CUDAQ_BACKEND_TENSORNET
__qpu__ void bell() {
  cudaq::qubit q, r;
  h(q);
  cx(q, r);
}

CUDAQ_TEST(GetStateTester, checkOverlapFromHostVector) {
  auto state = cudaq::get_state(bell);
  state.dump();
  std::vector<std::complex<double>> hostStateData{M_SQRT1_2, 0, 0, M_SQRT1_2};
  auto hostState = cudaq::state::from_data(hostStateData);
  hostState.dump();
  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState).real(), 1e-3);
}
#endif