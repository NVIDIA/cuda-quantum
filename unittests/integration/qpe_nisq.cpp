/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <stdio.h>

#include <cmath>

struct iqft {
  void operator()(cudaq::qview<> &q) __qpu__ {
    int N = q.size();
    // Swap qubits
    for (int i = 0; i < N / 2; ++i) {
      swap(q[i], q[N - i - 1]);
    }

    for (int i = 0; i < N - 1; ++i) {
      h(q[i]);
      int j = i + 1;
      for (int y = i; y >= 0; --y) {
        const double theta = -M_PI / std::pow(2.0, j - y);
        r1<cudaq::ctrl>(theta, q[j], q[y]);
      }
    }

    h(q[N - 1]);
  }
};

// Define an oracle
struct tgate {
  void operator()(cudaq::qubit &q) __qpu__ { t(q); }
};

struct qpe {
  template <typename StatePrep, typename Oracle>
  __qpu__ void operator()(const int n_c, const int n_q, StatePrep state_prep,
                          Oracle oracle) {
    // Allocate a register of qubits
    cudaq::qvector q(n_c + n_q);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(n_c);
    auto &state_register = q.back();

    // Prepare the eigenstate
    state_prep(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform ctrl-U^j
    for (int i = 0; i < n_c; ++i) {
      for (int j = 0; j < std::pow(2, i); ++j) {
        cudaq::control(oracle, {counting_qubits[i]}, state_register);
      }
    }

    // Apply inverse quantum fourier transform
    iqft{}(counting_qubits);

    // Measure to gather sampling statistics
    mz(counting_qubits);

    return;
  }
};

CUDAQ_TEST(QPENisqTester, checkSimple) {
  auto counts = cudaq::sample(
      qpe{}, 3, 1, [](cudaq::qubit &q) __qpu__ { x(q); }, tgate{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "100");
}

struct xOp {
  void operator()(cudaq::qubit &q) __qpu__ { x(q); }
};

struct qpeWithForwarding {
  __qpu__ void operator()(const int n_c, const int n_q, xOp &&state_prep,
                          tgate &&oracle) {
    // Allocate a register of qubits
    cudaq::qvector q(n_c + n_q);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(n_c);
    auto &state_register = q.back();

    // Prepare the eigenstate
    state_prep(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform ctrl-U^j
    for (int i = 0; i < n_c; ++i) {
      for (int j = 0; j < std::pow(2, i); ++j) {
        cudaq::control(oracle, {counting_qubits[i]}, state_register);
      }
    }

    // Apply inverse quantum fourier transform
    iqft{}(counting_qubits);

    // Measure to gather sampling statistics
    mz(counting_qubits);

    return;
  }
};

CUDAQ_TEST(QPENisqTester, checkPerfectForwardingBug) {
  auto counts = cudaq::sample(qpeWithForwarding{}, 3, 1, xOp{}, tgate{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "100");
}
