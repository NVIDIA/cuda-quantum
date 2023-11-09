/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/state.h"
#include "cudaq/builder/kernels.h"

inline cudaq::state
stateFromVector(const std::vector<std::complex<double>> &amplitudes) {
  return cudaq::state(std::make_tuple(
      std::vector<std::size_t>({amplitudes.size()}), amplitudes));
}

inline auto randomState(std::size_t numQUbits) {
  std::vector<std::complex<double>> stateVector(1ULL << numQUbits, 0.0);
  std::generate(stateVector.begin(), stateVector.end(), []() {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.5);
    return std::complex<double>(distribution(generator),
                                distribution(generator));
  });

  // Normalize the vector.
  double norm = 0;
  for (auto &amplitude : stateVector)
    norm += std::norm(amplitude);
  norm = std::sqrt(norm);
  std::transform(stateVector.begin(), stateVector.end(), stateVector.begin(),
                 [&norm](auto value) { return value / norm; });

  return stateVector;
}

CUDAQ_TEST(fromState, checkGetAlphaY) {
  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 2);
    std::vector<double> expected{1.57079633};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }

  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 1);
    std::vector<double> expected{0.0, 3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

CUDAQ_TEST(fromState, checkGetAlphaZ) {
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 3);
    std::vector<double> expected{1.17809725};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 2);
    std::vector<double> expected{0., 0.78539816};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }

  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 1);
    std::vector<double> expected{0., 0., 1.57079633, -3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(fromState, trivialStates) {
  constexpr auto maxNumQubits = 5u;
  for (auto n = 0u; n < maxNumQubits; ++n) {
    const auto dimension = (1ULL << (n + 1));
    for (auto i = 0u; i < dimension; ++i) {
      auto kernel = cudaq::make_kernel();
      auto qubits = kernel.qalloc(n + 1);

      std::vector<std::complex<double>> state(dimension, 0.);
      state[i] = 1.;
      cudaq::from_state(kernel, qubits, state);
      cudaq::state result = cudaq::get_state(kernel);
      cudaq::state expected = stateFromVector(state);
      // We test for overlap (i.e., fidelity) because the state vectors will
      // hardly match due to the approximative nature of the state preparation
      // algorithm.
      EXPECT_NEAR(result.overlap(expected), 1.0, 1e-6);
    }
  }
}

CUDAQ_TEST(fromState, checkQubitOrdering) {
  // In CUDA Quantum, we write numbers from lsb to msb. Thus, 001 means 4
  // instead of 1.
  auto toString = [](unsigned value, unsigned numBits) {
    std::string number;
    for (auto i = 0u; i < numBits; ++i)
      number.push_back(value & (1 << i) ? '1' : '0');
    std::reverse(number.begin(), number.end());
    return number;
  };

  constexpr auto maxNumQubits = 5u;
  for (auto n = 1u; n <= maxNumQubits; ++n) {
    const auto dimension = (1ULL << n);
    for (auto i = 0u; i < dimension; ++i) {
      auto kernel = cudaq::make_kernel();
      auto qubits = kernel.qalloc(n);

      std::vector<std::complex<double>> state(dimension, 0.);
      state[i] = 1.;
      cudaq::from_state(kernel, qubits, state);
      auto result = cudaq::sample(kernel);
      EXPECT_EQ(result.most_probable(), toString(i, n));
    }
  }
}

CUDAQ_TEST(fromState, randomStates) {
  constexpr auto maxNumQubits = 5u;
  for (auto n = 1u; n <= maxNumQubits; ++n) {
    const auto dimension = (1ULL << n);
    for (auto i = 0u; i < dimension; ++i) {
      auto kernel = cudaq::make_kernel();
      auto qubits = kernel.qalloc(n);

      auto state = randomState(n);
      cudaq::from_state(kernel, qubits, state);
      cudaq::state result = cudaq::get_state(kernel);
      cudaq::state expected = stateFromVector(state);
      // We test for overlap (i.e., fidelity) because the state vectors will
      // hardly match due to the approximative nature of the state preparation
      // algorithm.
      EXPECT_NEAR(result.overlap(expected), 1.0, 1e-6);
    }
  }
}

#endif
