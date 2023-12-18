/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <complex>
#include <gtest/gtest.h>
#include <iostream>
#include <math.h>

#include "CUDAQTestUtils.h"
#include "QppDMCircuitSimulator.cpp"

void print_state(const qpp::cmat &densityMatrix) {
  std::cout << "state = [";
  auto rows = densityMatrix.rows();
  auto cols = densityMatrix.cols();
  for (auto rdx = 0; rdx < rows; rdx++) {
    std::cout << "\n[ ";
    for (auto cdx = 0; cdx < cols; cdx++) {
      std::cout << densityMatrix(rdx, cdx) << " ";
    }
    std::cout << "]\n";
  }
  std::cout << "]\n";
}

qpp::ket getZeroState(const int numQubits) {
  qpp::idx state_dim = 1ULL << numQubits;
  qpp::ket zero_state = qpp::ket::Zero(state_dim);
  zero_state(0) = 1.0;
  return zero_state;
}

qpp::ket getOneState(const int numQubits) {
  qpp::idx state_dim = 1ULL << numQubits;
  qpp::ket one_state = qpp::ket::Zero(state_dim);
  one_state(state_dim - 1) = 1.0;
  return one_state;
}

qpp::cmat getZeroDensityMatrix(const int numQubits) {
  auto zeroVector = getZeroState(numQubits);
  // rho = |0> <0|
  return zeroVector * zeroVector.transpose();
}

qpp::cmat getOneDensityMatrix(const int numQubits) {
  auto oneVector = getOneState(numQubits);
  // rho = |1> <1|
  return oneVector * oneVector.transpose();
}

std::string getSampledBitString(QppNoiseCircuitSimulator &qppBackend,
                                std::vector<std::size_t> &&qubits) {
  std::cout << "sampling on the density matrix backend.\n";
  // Call `sample` and return the bitstring as the first element of the
  // measurement count map.
  cudaq::ExecutionContext ctx("sample", 1);
  qppBackend.setExecutionContext(&ctx);
  qppBackend.resetExecutionContext();
  auto sampleResults = ctx.result;
  return sampleResults.begin()->first;
}

// Tests for a previous bug in the density simulator, where
// the qubit ordering flipped after resizing the density matrix
// with new qubits.
CUDAQ_TEST(QPPTester, checkDensityOrderingBug) {
  {
    // Initialize QPP Backend 1 qubit at a time.
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q0));

    // Rotate to |1>
    qppBackend.x(q0);
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Add another qubit. Individually, should be |0>.
    auto q1 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q1));

    std::string got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ("10", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
  }

  {
    // Initialize QPP Backend with 2 qubits.
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Rotate both to |1>.
    qppBackend.x(q0);
    EXPECT_EQ(1, qppBackend.mz(q0));
    qppBackend.x(q1);
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Add another qubit. Individually, should be |0>.
    auto q2 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q2));

    std::string got_bitstring = getSampledBitString(qppBackend, {0, 1, 2});
    EXPECT_EQ("110", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));
    EXPECT_EQ(0, qppBackend.mz(q2));

    // Resize again with another new qubit.
    auto q3 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q3));

    // // Apply more rotations to the qubits as extra checks.
    qppBackend.x(q0);
    qppBackend.x(q1);
    qppBackend.x(q2);
    got_bitstring = getSampledBitString(qppBackend, {0, 1, 2, 3});
    EXPECT_EQ("0010", got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    EXPECT_EQ(1, qppBackend.mz(q2));
    EXPECT_EQ(0, qppBackend.mz(q3));
  }
}