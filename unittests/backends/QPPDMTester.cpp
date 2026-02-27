/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QppDMCircuitSimulator.cpp"

#include <gtest/gtest.h>
#include <iostream>

#include "CUDAQTestUtils.h"
#include "backends/QPPTester.h"

using QppNoiseSimulator =
    QppCircuitSimulatorTester<nvqir::QppCircuitSimulator<qpp::cmat>>;

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

// Tests for a previous bug in the density simulator, where
// the qubit ordering flipped after resizing the density matrix
// with new qubits.
CUDAQ_TEST(QPPTester, checkDensityOrderingBug) {
  {
    // Initialize QPP Backend 1 qubit at a time.
    QppNoiseSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q0));

    // Rotate to |1>
    qppBackend.x(q0);
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Add another qubit. Individually, should be |0>.
    auto q1 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q1));

    std::string got_bitstring = qppBackend.getSampledBitString({0, 1});
    EXPECT_EQ("10", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
  }

  {
    // Initialize QPP Backend with 2 qubits.
    QppNoiseSimulator qppBackend;
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

    std::string got_bitstring = qppBackend.getSampledBitString({0, 1, 2});
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
    got_bitstring = qppBackend.getSampledBitString({0, 1, 2, 3});
    EXPECT_EQ("0010", got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    EXPECT_EQ(1, qppBackend.mz(q2));
    EXPECT_EQ(0, qppBackend.mz(q3));
  }
}
