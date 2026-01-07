/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ --target=remote-mqpu %s -o %t && CUDAQ_BYPASS_PHASE_FOLDING_MINS=1 %t
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/resource_estimation.h>

#define ASSERT_NEAR(x,y,tolerance) assert(abs(x-y) < tolerance)

void check1() {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);

    x<cudaq::ctrl>(q[0], q[1]);
    rz(1.0, q[1]);

    auto &r = q.slice(1,1).front();
    h(r);
    rz(2.0, q[1]);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  auto state1 = cudaq::get_state(kernel);
  auto counts1 = cudaq::estimate_resources(kernel);
  assert(counts1.count("rz") == 2);
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  auto state2 = cudaq::get_state(kernel);
  auto counts2 = cudaq::estimate_resources(kernel);
  assert(counts2.count("rz") == 2);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

void check2() {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(5);

    x<cudaq::ctrl>(q[0],q[1]);
    rz(1.0, q[1]);

    for (int i = 0; i < 5; i++) {
      cudaq::qubit &r = q[i];
      h(r);
    }
  
    rz(1.0, q[1]);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  auto state1 = cudaq::get_state(kernel);
  auto counts1 = cudaq::estimate_resources(kernel);
  assert(counts1.count("rz") == 2);
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  auto state2 = cudaq::get_state(kernel);
  auto counts2 = cudaq::estimate_resources(kernel);
  assert(counts2.count("rz") == 2);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

auto subkernel(cudaq::qubit &r) __qpu__ {
  h(r);
}

void check3() {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);
    x<cudaq::ctrl>(q[0],q[1]);
    rz(1.0, q[1]);
    subkernel(q.back());
    rz(1.0, q[1]);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  auto state1 = cudaq::get_state(kernel);
  auto counts1 = cudaq::estimate_resources(kernel);
  assert(counts1.count("rz") == 2);
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  auto state2 = cudaq::get_state(kernel);
  auto counts2 = cudaq::estimate_resources(kernel);
  assert(counts2.count("rz") == 2);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

void check4(int seed) {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(3);
    
    x<cudaq::ctrl>(q[0],q[2]);
    rz(1.0, q[0]);
    rz(1.0, q[2]);

    h(q[1]);
    if (mz(q[1]))
        subkernel(q.front());
    else
        subkernel(q.back());

    rz(1.0, q[0]);
    rz(1.0, q[2]);
  };

  // need to make sure each kernel execution takes the same
  // execution path (side of the branch)
  cudaq::set_random_seed(seed);
  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  auto state1 = cudaq::get_state(kernel);
  auto counts1 = cudaq::estimate_resources(kernel);
  assert(counts1.count("rz") == 4);
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  auto state2 = cudaq::get_state(kernel);
  auto counts2 = cudaq::estimate_resources(kernel);
  assert(counts2.count("rz") == 4);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

int main() {
  check1();
  check2();
  check3();
  // manually validated at the time of test creation
  // that seed 1 and 2 takes different sides of the branching
  check4(1);
  check4(2);
}