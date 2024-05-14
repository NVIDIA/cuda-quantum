/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <iostream>
struct trotter {
  void operator()(cudaq::state initial_state, cudaq::spin_op ham,
                  double dt) __qpu__ {
    cudaq::qvector q(initial_state);
    ham.for_each_term([&](cudaq::spin_op &term) {
      const auto coeff = term.get_coefficient();
      const auto pauliWord = term.to_string(false);
      const double theta = coeff.real() * dt;
      cudaq::exp_pauli(dt, q, pauliWord.c_str());
    });
  }
};

int main() {
  const double g = 0.0;
  const double Jx = 1.0;
  const double Jy = 1.0;
  const double Jz = g;
  const double dt = 0.05;
  const int n_steps = 100;
  const int n_spins = 7;
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

  // Observe
  cudaq::spin_op average_magnetization(n_spins);
  for (int i = 0; i < n_spins; ++i)
    average_magnetization += ((1.0 / n_spins) * cudaq::spin::z(i));

  // Run loop
  std::vector<cudaq::complex> initial_state(1ULL << n_spins, 0.0);
  initial_state[0] = 1.0;
  auto state = cudaq::state::from_data(initial_state);
  std::vector<double> expResults;
  std::vector<double> runtimeMs;
  for (int i = 0; i < n_steps; ++i) {
    const auto start = std::chrono::high_resolution_clock::now();
    auto ham = heisenbergModelHam(i * dt);
    auto magnetization_exp_val =
        cudaq::observe(trotter{}, average_magnetization, state, ham, dt);
    expResults.emplace_back(magnetization_exp_val.expectation());
    state = cudaq::get_state(trotter{}, state, ham, dt);
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
}
