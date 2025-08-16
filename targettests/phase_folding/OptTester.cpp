/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target=remote-mqpu %s -o %t && CUDAQ_LOG_LEVEL=info %t | FileCheck .

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <random>

#define ASSERT_NEAR(x,y,tolerance) assert(abs(x-y) < tolerance)

void checkSimple() {
  printf("Running simple check\n");
  // CHECK-LABEL: Running simple check
  auto kernel = []() __qpu__ {
    cudaq::qubit q, p, r;
    h(q);
    h(p);
    h(r);
    rz(1.0, p);
    rz(2.0, r);
    x<cudaq::ctrl>(p, q);
    rz(3.0, q);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    h(r);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    rz(4.0, p);
    h(q);
    h(p);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // First run without phase folding
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(20);
  auto state1 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(1.000000, 1)
  // CHECK: (apply) rz(2.000000, 2)
  // CHECK: (apply) rz(3.000000, 0)
  // CHECK: (apply) rz(4.000000, 1)
  // Now run with phase folding
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(20);
  auto state2 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(2.000000, 2)
  // CHECK: (apply) rz(3.000000, 0)
  // CHECK: (apply) rz(5.000000, 1)

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

__qpu__ void subkernel(cudaq::qubit &q, cudaq::qubit &p) {
  rz(1.0, p);
  x<cudaq::ctrl>(p, q);
  rz(3.0, q);
};

void checkSubkernel() {
  printf("Running subkernel check\n");
  // CHECK-LABEL: Running subkernel check
  auto kernel = [&]() __qpu__ {
    cudaq::qubit q, p, r;
    h(q);
    h(p);
    h(r);
    rz(2.0, r);
    subkernel(q, p);
    x<cudaq::ctrl>(p, q);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    h(r);
    x<cudaq::ctrl>(p, r);
    x<cudaq::ctrl>(q, p);
    rz(4.0, p);
    h(q);
    h(p);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(30);
  auto state1 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(2.000000, 2)
  // CHECK: (apply) rz(1.000000, 1)
  // CHECK: (apply) rz(3.000000, 0)
  // CHECK: (apply) rz(4.000000, 1)
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(30);
  auto state2 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(2.000000, 2)
  // CHECK: (apply) rz(3.000000, 0)
  // CHECK: (apply) rz(5.000000, 1)

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

void checkClassical1() {
  printf("Running classical check #1\n");
  // CHECK-LABEL: Running classical check #1
  auto kernel = [&]() __qpu__ {
    cudaq::qubit q, p, r;
    rz(1.0, p);
    x<cudaq::ctrl>(q, p);
    rz(1.0, q);
    h(r);
    auto f = 2.;
    if (mz(r))
      f += 3.;
    x<cudaq::ctrl>(q, p);
    rz(f, p);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(40);
  auto state1 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(1.000000, 1)
  // CHECK: (apply) rz(1.000000, 0)
  // CHECK: (apply) rz(5.000000, 1)
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(40);
  auto state2 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(1.000000, 0)
  // CHECK: (apply) rz(6.000000, 1)

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

void checkClassical2() {
  printf("Running classical check #2\n");
  // CHECK-LABEL: Running classical check #2
  auto kernel = [&]() __qpu__ {
    cudaq::qubit q, p;
    float fs[] = { 2., 2., 2., 2., 2. };
    for (auto i = 0; i < 5; i++) {
      rz(fs[i], q);
      x<cudaq::ctrl>(q, p);
      rz(fs[i], q);
    }
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  // Without phase folding
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(50);
  auto state1 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // CHECK: (apply) rz(2.000000, 0)
  // With phase folding
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(50);
  auto state2 = cudaq::get_state(kernel);
  // CHECK: (apply) rz(20.000000, 0)

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.000001);
  ASSERT_NEAR(result.imag(), 0, 0.000001);
}

int main() {
  checkSimple();
  checkSubkernel();
  checkClassical1();
  checkClassical2();
}
