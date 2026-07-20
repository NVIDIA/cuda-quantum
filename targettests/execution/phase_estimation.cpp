/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target quantinuum --emulate %s -o %t && \
// RUN: CUDAQ_DUMP_JIT_IR=1 %t &> %basename_t.ir && \
// RUN: FileCheck %s < %basename_t.ir
// RUN: rm -f %basename_t.ir

#include <cudaq.h>
#include <iostream>

// A pure device quantum kernel defined as a free function (cannot be called
// from host code).
__qpu__ void iqft(cudaq::qview<> q) {
  int N = q.size();
  // Swap qubits
  for (int i = 0; i < N / 2; ++i)
    swap(q[i], q[N - i - 1]);

  for (int i = 0; i < N - 1; ++i) {
    h(q[i]);
    int j = i + 1;
    for (int y = i; y >= 0; --y) {
      double denom = (1UL << (j - y));
      const double theta = -M_PI / denom;
      r1<cudaq::ctrl>(theta, q[j], q[y]);
    }
  }

  h(q[N - 1]);
}

// CUDA-Q kernel call operators can be templated on input CUDA-Q kernel
// expressions. Here we define a general Phase Estimation algorithm that is
// generic on the eigenstate preparation and unitary evolution steps.
struct qpe {

  // Define the CUDA-Q call expression to take user-specified eigenstate and
  // unitary evolution kernels, as well as the number of qubits in the counting
  // register and in the eigenstate register.
  template <typename StatePrep, typename Unitary>
  void operator()(const int nCountingQubits, StatePrep &&state_prep,
                  Unitary &&oracle) __qpu__ {
    // Allocate a register of qubits
    cudaq::qvector q(nCountingQubits + 1);

    // Extract sub-registers, one for the counting qubits another for the
    // eigenstate register
    auto counting_qubits = q.front(nCountingQubits);
    auto &state_register = q.back();

    // Prepare the eigenstate
    state_prep(state_register);

    // Put the counting register into uniform superposition
    h(counting_qubits);

    // Perform `ctrl-U^j`
    for (int i = 0; i < nCountingQubits; ++i)
      for (int j = 0; j < (1 << i); ++j)
        cudaq::control(oracle, counting_qubits[i], state_register);

    // Apply inverse quantum Fourier transform
    iqft(counting_qubits);

    // Measure to gather sampling statistics
    mz(counting_qubits);
  }
};

struct r1PiGate {
  void operator()(cudaq::qubit &q) __qpu__ { r1(1., q); }
};

int main() {
  int nQubits = 2;
  auto counts = cudaq::sample(
      qpe{}, nQubits, [](cudaq::qubit &q) __qpu__ { x(q); }, r1PiGate{});
  return 0; // don't leave pass/fail status up to the phase of the moon
}

// CHECK-NOT: __quantum__qis__r1__body
