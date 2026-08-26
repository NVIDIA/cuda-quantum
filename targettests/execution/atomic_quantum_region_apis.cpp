/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target density-matrix-cpu %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target density-matrix-cpu -c %s -o %t.o && \
// RUN:   nvq++ --target density-matrix-cpu %t.o -o %t.staged && \
// RUN:   %t.staged | FileCheck %s

#include <algorithm>
#include <cmath>
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>
#include <iostream>
#include <string>
#include <vector>

struct atomic_workload {
  void operator()(cudaq::qview<> q) __qpu__ __atomic_quantum_region__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
  }
};

struct plain_workload {
  void operator()(cudaq::qview<> q) __qpu__ {
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
  }
};

struct atomic_round_trip {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    atomic_workload{}(q);
    cudaq::adjoint(atomic_workload{}, q);
  }
};

struct plain_round_trip {
  void operator()() __qpu__ {
    cudaq::qarray<3> q;
    plain_workload{}(q);
    cudaq::adjoint(plain_workload{}, q);
  }
};

struct measured_atomic_round_trip {
  int operator()() __qpu__ {
    cudaq::qarray<3> q;
    atomic_workload{}(q);
    cudaq::adjoint(atomic_workload{}, q);
    int result = 0;
    if (mz(q[0]))
      result += 1;
    if (mz(q[1]))
      result += 2;
    if (mz(q[2]))
      result += 4;
    return result;
  }
};

struct measured_plain_round_trip {
  int operator()() __qpu__ {
    cudaq::qarray<3> q;
    plain_workload{}(q);
    cudaq::adjoint(plain_workload{}, q);
    int result = 0;
    if (mz(q[0]))
      result += 1;
    if (mz(q[1]))
      result += 2;
    if (mz(q[2]))
      result += 4;
    return result;
  }
};

static std::size_t countOccurrences(const std::string &text,
                                    const std::string &token) {
  std::size_t count = 0;
  std::size_t position = 0;
  while ((position = text.find(token, position)) != std::string::npos) {
    ++count;
    position += token.size();
  }
  return count;
}

int main() {
  constexpr std::size_t shots = 10;

  // An ideal U followed by U adjoint is the identity in both kernels. Apply a
  // deterministic XX error after every CNOT. The plain gates cancel before
  // noise insertion and leave |000>. Atomic regions preserve four CNOT error
  // sites, which drive the state to |011>.
  std::vector<cudaq::real> cnotErrorProbabilities(cudaq::pauli2::num_parameters,
                                                  0.0);
  cnotErrorProbabilities[4] = 1.0; // XX
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(
      cudaq::pauli2(cnotErrorProbabilities),
      /*numControls=*/1);

  // Sample
  const cudaq::sample_options sampleOptions{.shots = shots, .noise = noise};
  auto atomicCounts = cudaq::sample(sampleOptions, atomic_round_trip{});
  auto plainCounts = cudaq::sample(sampleOptions, plain_round_trip{});

  assert(atomicCounts.count("011") == shots);
  assert(plainCounts.count("000") == shots);

  // Observe
  const cudaq::observe_options observeOptions{.noise = noise};
  auto atomicObservation =
      cudaq::observe(observeOptions, atomic_round_trip{}, cudaq::spin_op::z(1));
  auto plainObservation =
      cudaq::observe(observeOptions, plain_round_trip{}, cudaq::spin_op::z(1));

  assert(std::abs(atomicObservation.expectation() + 1.0) < 1e-6);
  assert(std::abs(plainObservation.expectation() - 1.0) < 1e-6);

  // Run
  auto atomicValues = cudaq::run(shots, noise, measured_atomic_round_trip{});
  auto plainValues = cudaq::run(shots, noise, measured_plain_round_trip{});

  assert(std::count(atomicValues.begin(), atomicValues.end(), 6) == shots);
  assert(std::count(plainValues.begin(), plainValues.end(), 0) == shots);

  // Draw
  auto atomicDrawing = cudaq::contrib::draw(atomic_round_trip{});
  auto plainDrawing = cudaq::contrib::draw(plain_round_trip{});

  const std::string expectedAtomicDrawing =
      "     ╭───╮                    ╭───╮\n"
      "q0 : ┤ h ├──●──────────────●──┤ h ├\n"
      "     ╰───╯╭─┴─╮          ╭─┴─╮╰───╯\n"
      "q1 : ─────┤ x ├──●────●──┤ x ├─────\n"
      "          ╰───╯╭─┴─╮╭─┴─╮╰───╯     \n"
      "q2 : ──────────┤ x ├┤ x ├──────────\n"
      "               ╰───╯╰───╯          \n";
  assert(atomicDrawing == expectedAtomicDrawing);
  assert(plainDrawing.empty());

  printf("SUCCESS\n");
  return 0;
}

// CHECK: SUCCESS
