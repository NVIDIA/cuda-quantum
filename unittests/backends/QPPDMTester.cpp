/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QppDMCircuitSimulator.cpp"

#include <gtest/gtest.h>
#include <iostream>

#include "CUDAQTestUtils.h"

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

bool EXPECT_EQ_MATRIX(qpp::cmat want_cmat, qpp::cmat got_cmat,
                      double epsilon = 1e-6) {
  return ((want_cmat - got_cmat).norm() < epsilon);
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

CUDAQ_TEST(QPPTester, checkSetStateDensity) {
  {
    // Initialize QPP Backend with 1 qubit.
    const int num_qubits = 1;
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Assert that we're starting in the 0-state.
    qpp::cmat got_state = qppBackend.getDensityMatrix();
    qpp::cmat want_state = getZeroDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);

    // Test 1: Would like to manually set the state to the `|1><1|` state.
    std::vector<std::complex<double>> inputState{0., 0., 0., 1.};
    qppBackend.setStateData(inputState);

    got_state = qppBackend.getDensityMatrix();
    want_state = getOneDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);
    std::string got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ("1", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Test 2: Build up a custom density matrix as a flattened
    // vector of size `2^n * 2^n`.
    inputState = {};
    for (auto i = 0; i < pow(2, num_qubits) * pow(2, num_qubits); i++) {
      inputState.push_back(M_PI);
    }
    qppBackend.setStateData(inputState);

    want_state.fill(M_PI);
    got_state = qppBackend.getDensityMatrix();
    EXPECT_EQ(want_state, got_state);

    qppBackend.deallocate(q0);
  }

  {
    // Initialize QPP Backend with 2 qubits.
    const int num_qubits = 2;
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Assert that we're starting in the 0-state.
    qpp::cmat got_state = qppBackend.getDensityMatrix();
    qpp::cmat want_state = getZeroDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);

    // Test 1: Would like to manually set the state to the `|1><1|` state.
    std::vector<std::complex<double>> inputState{
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.};
    qppBackend.setStateData(inputState);

    got_state = qppBackend.getDensityMatrix();
    want_state = getOneDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);
    std::string got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ("11", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Test 2: Build up a custom density matrix as a flattened
    // vector of size `2^n * 2^n`.
    inputState = {};
    for (auto i = 0; i < pow(2, num_qubits) * pow(2, num_qubits); i++) {
      inputState.push_back(M_PI_2);
    }
    qppBackend.setStateData(inputState);

    want_state.fill(M_PI_2);
    got_state = qppBackend.getDensityMatrix();
    EXPECT_EQ(want_state, got_state);

    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  {
    // Initialize QPP Backend with 3 qubits.
    const int num_qubits = 3;
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();
    auto q2 = qppBackend.allocateQubit();

    // Assert that we're starting in the 0-state.
    qpp::cmat got_state = qppBackend.getDensityMatrix();
    qpp::cmat want_state = getZeroDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);

    // Test 1: Would like to manually set the state to the `|1><1|` state.
    std::vector<std::complex<double>> inputState(
        pow(2, num_qubits) * pow(2, num_qubits), 0.0);
    inputState.back() = 1.0;
    qppBackend.setStateData(inputState);

    got_state = qppBackend.getDensityMatrix();
    want_state = getOneDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);
    std::string got_bitstring = getSampledBitString(qppBackend, {0, 1, 2});
    EXPECT_EQ("111", got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));
    EXPECT_EQ(1, qppBackend.mz(q2));

    // Test 2: Build up a custom density matrix as a flattened
    // vector of size `2^n * 2^n`.
    inputState = {};
    for (auto i = 0; i < pow(2, num_qubits) * pow(2, num_qubits); i++) {
      inputState.push_back(M_PI_4);
    }
    qppBackend.setStateData(inputState);

    want_state.fill(M_PI_4);
    got_state = qppBackend.getDensityMatrix();
    EXPECT_EQ(want_state, got_state);

    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
    qppBackend.deallocate(q2);
  }

  // More advanced integration test.
  {
    // Initialize QPP Backend with 2 qubits initially.
    // Will add a third qubit later.
    int num_qubits = 2;
    QppNoiseCircuitSimulator qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Assert that we're starting in the 0-state.
    qpp::cmat got_state = qppBackend.getDensityMatrix();
    qpp::cmat want_state = getZeroDensityMatrix(num_qubits);
    EXPECT_EQ(want_state, got_state);

    // Building up the equivalent of a density matrix that has
    // undergone a Hadamard rotation.
    auto value = 1. / pow(2, num_qubits);
    std::vector<std::complex<double>> inputState(
        pow(2, num_qubits) * pow(2, num_qubits), value);
    qppBackend.setStateData(inputState);

    got_state = qppBackend.getDensityMatrix();
    want_state.fill(value);
    EXPECT_EQ(want_state, got_state);

    // Add a third qubit to the system, and ensure it is
    // in the |0> state, while the first two qubits remain
    // in the superposition state.
    num_qubits = 3;
    auto q2 = qppBackend.allocateQubit();
    EXPECT_EQ(0, qppBackend.mz(q2));

    // Kronecker a new, single qubit |0> state onto the
    // `want_state` matrix.
    want_state = qpp::kron(want_state, getZeroDensityMatrix(1));
    got_state = qppBackend.getDensityMatrix();
    EXPECT_EQ(want_state, got_state);

    // Apply Hadamard's via gates to the first 2 qubits and
    // assert that this produces the identity.
    qppBackend.h(q0);
    qppBackend.h(q1);
    got_state = qppBackend.getDensityMatrix();
    want_state = getZeroDensityMatrix(num_qubits);
    EXPECT_EQ_MATRIX(got_state, want_state, 1e-10);

    // Finally, rotate the third qubit to the |1> state to ensure
    // it may still be acted upon individually.
    qppBackend.x(q2);
    got_state = qppBackend.getDensityMatrix();
    // Have to build up our expected state manually as |0> x |0> x |1>
    want_state = qpp::kron(getZeroState(1), getZeroState(1));
    want_state = qpp::kron(want_state, getOneState(1));
    // `rho = |want_state> <want_state|`
    want_state = want_state * want_state.transpose();
    EXPECT_EQ_MATRIX(got_state, want_state, 1e-10);

    // Confirm that the bitstring returned from `::sample`
    // is `001` by running 1 shot of simulation.
    std::string got_bitstring = getSampledBitString(qppBackend, {0, 1, 2});
    std::cout << "measured " << got_bitstring << "\n";
    std::string want_bitstring = std::string("001");
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    EXPECT_EQ(1, qppBackend.mz(q2));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
    qppBackend.deallocate(q2);
  }
}
