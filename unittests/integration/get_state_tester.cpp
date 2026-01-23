/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

// State operations not supported in Stim.
#ifndef CUDAQ_BACKEND_STIM

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

  // Multiple amplitude access
  const auto amplitudes = state.amplitudes({{0, 0}, {1, 1}});
  EXPECT_EQ(amplitudes.size(), 2);
  EXPECT_NEAR(1. / std::sqrt(2.), amplitudes[0].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), amplitudes[1].real(), 1e-3);
#endif

  EXPECT_NEAR(state.overlap(state).real(), 1.0, 1e-3);

  auto kernelNoop = []() __qpu__ {
    cudaq::qubit q, r;
    x(q);
    x(q);
  };

// Check <00|Bell> = 1/sqrt(2)
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelNoop)).real(), (1. / 2.),
              1e-3);
#else
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelNoop)).real(),
              1. / std::sqrt(2.), 1e-3);
#endif

  auto kernelX = []() __qpu__ {
    cudaq::qubit q, r;
    x(q);
    x(r);
  };

// Check <11|Bell> = 1/sqrt(2)
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelX)).real(), 1. / 2., 1e-3);
#else
  EXPECT_NEAR(state.overlap(cudaq::get_state(kernelX)).real(),
              1. / std::sqrt(2.), 1e-3);
#endif

  // Check endianess of basis state
  auto kernel1 = []() __qpu__ {
    cudaq::qvector qvec(5);
    x(qvec[1]);
    x(qvec[2]);
  };

  auto state1 = cudaq::get_state(kernel1);
  EXPECT_NEAR(1.0, state1.amplitude({0, 1, 1, 0, 0}).real(), 1e-3);

  // Check sign of overlap calculation:
  // Basic Bell states: (|00> + |11>) and (|00> - |11>) are orthogonal states,
  // and should have zero overlap. These Bell states form a maximally entangled
  // basis, known as the Bell basis, of the Hilbert space.
  auto bell1 = []() __qpu__ {
    cudaq::qvector qvec(2);
    h(qvec[0]);
    cx(qvec[0], qvec[1]);
  };

  auto bellState1 = cudaq::get_state(bell1);

  auto bell2 = []() __qpu__ {
    cudaq::qvector qvec(2);
    x(qvec[0]);
    h(qvec[0]);
    cx(qvec[0], qvec[1]);
  };

  auto bellState2 = cudaq::get_state(bell2);
  const auto overlap = bellState1.overlap(bellState2);
  EXPECT_NEAR(0.0, overlap.real(), 1e-3);

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
  std::vector<cudaq::complex> hostStateData{M_SQRT1_2, 0, 0, M_SQRT1_2};
  auto hostState = cudaq::state::from_data(hostStateData);
  hostState.dump();
  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState).real(), 1e-3);
}
#endif

CUDAQ_TEST(GetStateTester, checkKron) {
  auto force_kron = [](const std::vector<std::complex<cudaq::real>> &vec)
                        __qpu__ {
                          cudaq::qubit a;
                          cudaq::qvector qvec(cudaq::state{vec});
                        };
  // Construct a 6-qubit |111111> state
  const int num_qubits_input_state = 6;
  std::vector<std::complex<cudaq::real>> hostStateData(
      1 << num_qubits_input_state);
  hostStateData[hostStateData.size() - 1] = 1.0;

  auto counts = cudaq::sample(force_kron, hostStateData);

  // Expect a single state with a deterministic outcome
  EXPECT_EQ(counts.size(), 1);
  EXPECT_EQ(counts.begin()->first,
            "0" + std::string(num_qubits_input_state, '1'));
}

#endif
