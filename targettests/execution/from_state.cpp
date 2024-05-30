/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: nvq++ %cpp_std %s -o %t && %t | FileCheck %s

// //include <cudaq.h>

// __qpu__ void test(std::vector<cudaq::complex> inState) {
//   cudaq::qvector q = inState;
// }

// // CHECK: size 2

// int main() {
//   std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2};
//   auto counts = cudaq::sample(test, vec);
//   counts.dump();

//   printf("size %zu\n", counts.size());


// }

#include <cudaq.h>
#include <iostream>

// Alternating up/down spins
struct initState {
  void operator()(int num_spins) __qpu__ {
    cudaq::qvector q(num_spins);
    for (int qId = 0; qId < num_spins; qId += 2)
      x(q[qId]);
  }
};

struct trotter {
  // Note: This performs a single-step Trotter on top of an initial state, e.g.,
  // result state of the previous Trotter step.
  void operator()(cudaq::state initial_state, std::vector<double> data, int n_spins, double dt) __qpu__ {
    cudaq::qvector q(initial_state);
    cudaq::spin_op ham(data, n_spins);
    ham.for_each_term([&](cudaq::spin_op &term) {
      const auto coeff = term.get_coefficient();
      const auto pauliWord = term.to_string(false);
      const double theta = coeff.real() * dt;
      cudaq::exp_pauli(dt, q, pauliWord.c_str());
    });
  }
};

int main() {
  const double g = 1.0;
  const double Jx = 1.0;
  const double Jy = 1.0;
  const double Jz = g;
  const double dt = 0.05;
  const int n_steps = 100;
  const int n_spins = 25;
  const double omega = 2 * M_PI;
  const auto heisenbergModelHam = [&](double t) -> cudaq::spin_op {
    cudaq::spin_op tdOp(n_spins);
    for (int i = 0; i < n_spins - 1; ++i) {
      tdOp += (Jx * cudaq::spin::x(i) * cudaq::spin::x(i + 1));
      tdOp += (Jy * cudaq::spin::y(i) * cudaq::spin::y(i + 1));
      tdOp += (Jz * cudaq::spin::z(i) * cudaq::spin::z(i + 1));
    }
    for (int i = 0; i < n_spins; ++i)
      tdOp += (std::cos(omega * t) * cudaq::spin::x(i));
    return tdOp;
  };
  // Observe the average magnetization of all spins (<Z>)
  cudaq::spin_op average_magnetization(n_spins);
  for (int i = 0; i < n_spins; ++i)
    average_magnetization += ((1.0 / n_spins) * cudaq::spin::z(i));
  average_magnetization -= 1.0;
  
  // Run loop
  auto state = cudaq::get_state(initState{}, n_spins);
  std::vector<double> expResults;
  std::vector<double> runtimeMs;
  for (int i = 0; i < n_steps; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    auto ham = heisenbergModelHam(i * dt);
    auto data = ham.getDataRepresentation();
    auto magnetization_exp_val =
        cudaq::observe(trotter{}, average_magnetization, state, data, n_spins, dt);
    expResults.emplace_back(magnetization_exp_val.expectation());
    state = cudaq::get_state(trotter{}, state, data, n_spins, dt);
    const auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    runtimeMs.emplace_back(duration.count()/1000.0);
  }
  std::cout << "Runtime [ms]: [";
  for (const auto &x : runtimeMs)
    std::cout << x << ", ";
  std::cout << "]\n";
  return 0;

  std::cout << "hello" << std::endl;
}
