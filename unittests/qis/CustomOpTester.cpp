/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"

#include <cmath>
#include <complex>

using namespace std::complex_literals;

cudaq_register_operation(custom_h,
                         (std::vector<std::vector<std::complex<double>>>{
                             {M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}}));

cudaq_register_operation(
    custom_x, (std::vector<std::vector<std::complex<double>>>{{0, 1}, {1, 0}}));

cudaq_register_operation(custom_s,
                         (std::vector<std::vector<std::complex<double>>>{
                             {1, 0}, {0, 1i}}));

cudaq_register_operation(
    custom_cnot_be,
    (std::vector<std::vector<std::complex<double>>>{
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}}));

cudaq_register_operation(
    custom_cnot_le,
    (std::vector<std::vector<std::complex<double>>>{
        {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}}));

cudaq_register_operation(toffoli_be,
                         (std::vector<std::vector<std::complex<double>>>{
                             {1, 0, 0, 0, 0, 0, 0, 0},
                             {0, 1, 0, 0, 0, 0, 0, 0},
                             {0, 0, 1, 0, 0, 0, 0, 0},
                             {0, 0, 0, 1, 0, 0, 0, 0},
                             {0, 0, 0, 0, 1, 0, 0, 0},
                             {0, 0, 0, 0, 0, 1, 0, 0},
                             {0, 0, 0, 0, 0, 0, 0, 1},
                             {0, 0, 0, 0, 0, 0, 1, 0}}));

CUDAQ_TEST(CustomOpTester, checkBasic) {
  auto use_custom_op = []() {
    cudaq::qvector qubits(2);
    custom_h(qubits[0]);
    custom_x<cudaq::ctrl>(qubits[0], qubits[1]);
  };
  auto counts = cudaq::sample(use_custom_op);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
}

CUDAQ_TEST(CustomOpTester, checkAdjoint) {
  auto kernel = []() {
    auto q = cudaq::qubit();
    h(q);
    custom_s<cudaq::adj>(q);
    custom_s<cudaq::adj>(q);
    h(q);
    mz(q);
  };
  auto counts = cudaq::sample(kernel);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "1");
  }
}

CUDAQ_TEST(CustomOpTester, checkTwoQubitOp) {
  auto bell_pair_be = []() {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    custom_cnot_be(qubits[1], qubits[0]); // custom_cnot_be(target, control)
  };
  auto counts = cudaq::sample(bell_pair_be);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
  }

  auto bell_pair_le = []() {
    cudaq::qvector qubits(2);
    h(qubits[0]);
    custom_cnot_le(qubits[0], qubits[1]); // custom_cnot_le(control, target)
  };
  counts = cudaq::sample(bell_pair_le);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "00" || bits == "11");
  }
}

#ifndef CUDAQ_BACKEND_TENSORNET_MPS
// SKIP_TEST: Reason - "Gates on 3 or more qubits are unsupported."
CUDAQ_TEST(CustomOpTester, checkToffoli) {
  auto test_toffoli_be = []() {
    cudaq::qvector q(3);
    x(q);
    toffoli_be(q[0], q[1], q[2]); // q[0] is the target
  };
  auto counts = cudaq::sample(test_toffoli_be);
  counts.dump();
  for (auto &[bits, count] : counts) {
    EXPECT_TRUE(bits == "011");
  }
}
#endif