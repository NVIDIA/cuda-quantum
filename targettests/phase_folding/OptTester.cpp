/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target=remote-mqpu %s -o %t && %t

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <random>

#define ASSERT_NEAR(x,y,tolerance) assert(abs(x-y) < tolerance)

void checkSimple() {
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
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(20);
  auto state1 = cudaq::get_state(kernel);
  //state1.dump();
  // Add resource counter call (once merged) here to make sure opts are actually
  // done
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(20);
  auto state2 = cudaq::get_state(kernel);
  //state2.dump();

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.0000001);
  ASSERT_NEAR(result.imag(), 0, 0.0000001);
}

__qpu__ void subkernel(cudaq::qubit &q, cudaq::qubit &p) {
  rz(1.0, p);
  x<cudaq::ctrl>(p, q);
  rz(3.0, q);
};

void checkSubkernel() {
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
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(30);
  auto state1 = cudaq::get_state(kernel);
  // Add resource counter call (once merged) here to make sure opts are actually
  // done
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(30);
  auto state2 = cudaq::get_state(kernel);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.0000001);
  ASSERT_NEAR(result.imag(), 0, 0.0000001);
}

void checkClassicalSimple() {
  auto kernel = [&]() __qpu__ {
    cudaq::qubit q, p, r;
    x<cudaq::ctrl>(q, p);
    rz(1.0, q);
    rz(1.0, p);
    h(r);
    auto f = 3.1415 / 2.;
    if (mz(r))
      f += 3.1415;
    x<cudaq::ctrl>(q, p);
    rz(f, p);
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(40);
  auto state1 = cudaq::get_state(kernel);
  // Add resource counter call (once merged) here to make sure opts are actually
  // done
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(40);
  auto state2 = cudaq::get_state(kernel);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.0000001);
  ASSERT_NEAR(result.imag(), 0, 0.0000001);
}

void checkClassicalComplex() {
  auto kernel = [&]() __qpu__ {
    cudaq::qubit q, p;
    float fs[] = { 2., 2., 2., 2., 2., 2., 2., 2., 2., 2. };
    for (auto i = 0; i < 10; i++) {
      rz(fs[i], q);
      x<cudaq::ctrl>(q, p);
      rz(fs[i], q);
    }
  };

  const auto PHASE_SWITCH = "CUDAQ_PHASE_FOLDING";
  setenv(PHASE_SWITCH, "0", true);
  cudaq::set_random_seed(50);
  auto state1 = cudaq::get_state(kernel);
  // Add resource counter call (once merged) here to make sure opts are actually
  // done
  setenv(PHASE_SWITCH, "1", true);
  cudaq::set_random_seed(50);
  auto state2 = cudaq::get_state(kernel);

  assert(state1.get_num_qubits() == state2.get_num_qubits());
  auto result = state1.overlap(state2);
  ASSERT_NEAR(result.real(), 1, 0.0000001);
  ASSERT_NEAR(result.imag(), 0, 0.0000001);
}

int main() {
  checkSimple();
  checkSubkernel();
  checkClassicalSimple();
  checkClassicalComplex();
}
