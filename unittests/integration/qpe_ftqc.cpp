/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
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

struct qpe {
  double operator()(const int n_c, const int n_q) __qpu__ {
    // Allocate a register of qubits
    cudaq::qvector q(n_c + n_q);

    // Extract sub-registers, one for the counting qubits
    // another for the eigenstate register
    auto counting_qubits = q.front(n_c);

    auto &state_register = q.back();

    // Prepare the eigenstate
    x(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform ctrl-U^j
    for (int i = 0; i < n_c; ++i) {
      for (int j = 0; j < std::pow(2, i); ++j) {
        t<cudaq::ctrl>(counting_qubits[i], state_register);
      }
    }

    // Apply inverse quantum fourier transform
    iqft{}(counting_qubits);
    // Measure and compute the phase...
    return cudaq::to_integer(mz(counting_qubits)) / std::pow(2, n_c);
  }
};

CUDAQ_TEST(QPEFTQCTester, checkSimple) {
  double phase = qpe{}(3, 1);
  EXPECT_NEAR(phase, .125, 1e-4);
  printf("Phase = %lf\n", phase);
}
