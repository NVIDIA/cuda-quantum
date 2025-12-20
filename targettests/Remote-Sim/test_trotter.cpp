/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// TODO-FIX-KERNEL-EXEC
// RUN: nvq++ --enable-mlir --target remote-mqpu -fkernel-exec-kind=2 %s -o %t  && %t | FileCheck %s
// clang-format on

#include <complex>
#include <cudaq.h>
#include <iostream>
#include <string>

// Compute magnetization using Suzuki-Trotter approximation.
// This example demonstrates usage of quantum states in kernel mode.
//
// Details
// https://pubs.aip.org/aip/jmp/article-abstract/32/2/400/229229/General-theory-of-fractal-path-integrals-with
//
// Hamiltonian used
// https://en.m.wikipedia.org/wiki/Quantum_Heisenberg_model

// If you have a NVIDIA GPU you can use this example to see
// that the GPU-accelerated backends can easily handle a
// larger number of qubits compared the CPU-only backend.
//
// Depending on the available memory on your GPU, you can
// set the number of qubits to around 30 qubits, and run
// the execution command with `-target nvidia` option.
//
// Note: Without setting the target to the `nvidia` backend,
// there will be a noticeable decrease in simulation performance.
// This is because the CPU-only backend has difficulty handling
// 30+ qubit simulations.

int SPINS = 11; // set to around 25 qubits for `nvidia` target
int STEPS = 10; // set to around 100 for `nvidia` target

// Compile and run with:
// clang-format off
// ```
// nvq++ --enable-mlir -v trotter_kernel_mode.cpp -o trotter.x --target nvidia && ./trotter.x
// ```
// clang-format off

// Alternating up/down spins
struct initState {
  void operator()(int num_spins) __qpu__ {
    cudaq::qvector q(num_spins);
    for (int qId = 0; qId < num_spins; qId += 2)
      x(q[qId]);
  }
};

std::vector<double> term_coefficients(cudaq::spin_op op) {
  std::vector<double> result{};
  for (const auto &term : op) {
    const auto coeff = term.evaluate_coefficient().real();
    result.push_back(coeff);
  }
  return result;
}

std::vector<cudaq::pauli_word> pauli_words(cudaq::spin_op op) {
  // The kernel code above is written to assume that each term acts on
  // all qubits. We hence expand each pauli word to extend to all qubits.
  auto nr_qubits = op.num_qubits(); // relying on qubits being consecutive
  std::vector<cudaq::pauli_word> result{};
  for (const auto &term : op)
    result.push_back(term.get_pauli_word(nr_qubits));
  return result;
}

struct trotter {
  // Note: This performs a single-step Trotter on top of an initial state, e.g.,
  // result state of the previous Trotter step.
  void operator()(cudaq::state *initial_state,
                  std::vector<double> &coefficients,
                  std::vector<cudaq::pauli_word> &words, double dt) __qpu__ {
    cudaq::qvector q(initial_state);
    for (std::size_t i = 0; i < coefficients.size(); ++i) {
      cudaq::exp_pauli(coefficients[i] * dt, q, words[i]);
    }
  }
};

int run_steps(int steps, int spins) {
  const double g = 1.0;
  const double Jx = 1.0;
  const double Jy = 1.0;
  const double Jz = g;
  const double dt = 0.05;
  const int n_steps = steps;
  const std::size_t n_spins = spins;
  const double omega = 2 * M_PI;
  const auto heisenbergModelHam = [&](double t) -> cudaq::spin_op {
    cudaq::spin_op tdOp(n_spins);
    for (std::size_t i = 0; i < n_spins - 1; ++i) {
      tdOp += (Jx * cudaq::spin_op::x(i) * cudaq::spin_op::x(i + 1));
      tdOp += (Jy * cudaq::spin_op::y(i) * cudaq::spin_op::y(i + 1));
      tdOp += (Jz * cudaq::spin_op::z(i) * cudaq::spin_op::z(i + 1));
    }
    for (std::size_t i = 0; i < n_spins; ++i)
      tdOp += (std::cos(omega * t) * cudaq::spin_op::x(i));
    return tdOp;
  };
  // Observe the average magnetization of all spins (<Z>)
  cudaq::spin_op average_magnetization(n_spins);
  for (std::size_t i = 0; i < n_spins; ++i)
    average_magnetization += ((1.0 / n_spins) * cudaq::spin_op::z(i));

  // Run loop
  auto state = cudaq::get_state(initState{}, n_spins);
  std::vector<double> expResults;
  std::vector<double> runtimeMs;
  for (int i = 0; i < n_steps; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    auto ham = heisenbergModelHam(i * dt);
    auto coefficients = term_coefficients(ham);
    auto words = pauli_words(ham);
    auto magnetization_exp_val = cudaq::observe(
        trotter{}, average_magnetization, &state, coefficients, words, dt);
    auto result = magnetization_exp_val.expectation();
    expResults.emplace_back(result);
    state = cudaq::get_state(trotter{}, &state, coefficients, words, dt);
    const auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    auto timeInSeconds = duration.count() / 1000.0 / 1000.0;
    runtimeMs.emplace_back(timeInSeconds);
    std::cout << "Step " << i << ": time [s]: " << timeInSeconds
              << ", result: " << result << std::endl;
  }
  std::cout << std::endl;

  // Print runtimes and results (useful for plotting).
  std::cout << "Step times [s]: [";
  for (const auto &x : runtimeMs)
    std::cout << x << ", ";
  std::cout << "]" << std::endl;

  std::cout << "Results: [";
  for (const auto &x : expResults)
    std::cout << x << ", ";
  std::cout << "]" << std::endl;

  std::cout << std::endl;
  return 0;
}

int main() {
  const auto start = std::chrono::high_resolution_clock::now();
  run_steps(STEPS, SPINS);
  const auto stop = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "Total running time: " << duration.count() / 1000.0 / 1000.0
            << "s" << std::endl;
}

// CHECK:  Step 0: time [s]: [[t0:.*]], result: [[v0:.*]]
// CHECK:  Step 1: time [s]: [[t1:.*]], result: [[v1:.*]]
// CHECK:  Step 2: time [s]: [[t2:.*]], result: [[v2:.*]]
// CHECK:  Step 3: time [s]: [[t3:.*]], result: [[v3:.*]]
// CHECK:  Step 4: time [s]: [[t4:.*]], result: [[v4:.*]]
// CHECK:  Step 5: time [s]: [[t5:.*]], result: [[v5:.*]]
// CHECK:  Step 6: time [s]: [[t6:.*]], result: [[v6:.*]]
// CHECK:  Step 7: time [s]: [[t7:.*]], result: [[v7:.*]]
// CHECK:  Step 8: time [s]: [[t8:.*]], result: [[v8:.*]]
// CHECK:  Step 9: time [s]: [[t9:.*]], result: [[v9:.*]]

// CHECK:  Step times [s]: [[ts:.*]]
// CHECK:  Results: [[rs:.*]]

// CHECK:  Total running time: [[tts:.*]]s
