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
#include "QppCircuitSimulator.cpp"

#define _USE_MATH_DEFINES

using namespace nvqir;

void print_state(const qpp::ket &stateVector) {
  std::cout << "state = [";
  for (const auto &term : stateVector) {
    std::cout << term << " ";
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

std::string getSampledBitString(QppCircuitSimulator<qpp::ket> &qppBackend,
                                std::vector<std::size_t> &&qubits) {
  // Call `sample` and return the bitstring as the first element of the
  // measurement count map.
  cudaq::ExecutionContext ctx("sample", 1);
  qppBackend.setExecutionContext(&ctx);
  qppBackend.resetExecutionContext();
  auto sampleResults = ctx.result;
  return sampleResults.begin()->first;
}

// Helper function for comparing two complex state vectors up to a certain
// tolerance.
bool EXPECT_EQ_KETS(qpp::ket want_ket, qpp::ket got_ket,
                    double epsilon = 1e-6) {
  assert(want_ket.size() == got_ket.size() &&
         "vectors must be of equal size for comparison");
  double want_real, got_real;
  std::complex<double> want_imag, got_imag;
  for (int i = 0; i < want_ket.size(); i++) {
    want_real = static_cast<double>(want_ket(i).real());
    got_real = static_cast<double>(got_ket(i).real());
    want_imag = static_cast<std::complex<double>>(want_ket(i).imag());
    got_imag = static_cast<std::complex<double>>(got_ket(i).imag());
    EXPECT_NEAR(want_real, got_real, epsilon);
    assert(std::abs(want_imag - got_imag) < epsilon);
  }
  return true;
}

// Checks that we're initializing the backend to the expected
// state and confirms via calls to `measure` and `sample`, as
// well as by checking the state vector.
CUDAQ_TEST(QPPTester, checkInitialState) {
  // Keeping track of the state vector.
  qpp::ket got_state;
  qpp::ket want_state;
  std::string got_bitstring;
  std::string want_bitstring;
  // Initialize QPP Backend with 2 qubits
  const int num_qubits = 2;
  QppCircuitSimulator<qpp::ket> qppBackend;
  auto q0 = qppBackend.allocateQubit();
  auto q1 = qppBackend.allocateQubit();

  // Assert that we're starting in the 0-state.
  got_state = qppBackend.getStateVector();
  want_state = getZeroState(num_qubits);
  EXPECT_EQ(want_state, got_state);

  // Confirm that calling measure on each qubit
  // also returns the 0 state.
  EXPECT_EQ(0, qppBackend.mz(q0));
  EXPECT_EQ(0, qppBackend.mz(q1));

  // Confirm that the bitstring returned from `::sample`
  // is `00` by running 1 shot of simulation.
  got_bitstring = getSampledBitString(qppBackend, {0, 1});
  want_bitstring = std::string("00");
  EXPECT_EQ(want_bitstring, got_bitstring);
  qppBackend.deallocate(q0);
  qppBackend.deallocate(q1);
}

// Testing the accuracy of all non-parameterized single-qubit
// gates.
CUDAQ_TEST(QPPTester, checkSingleQGates) {

  // Checking the single qubit X-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Place just q0 in the 1-state with X-gate.
    qppBackend.x(q0);
    // State vector should now be `|1> <0| = (0 0 1 0)`.
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // Confirm state vectors.
    EXPECT_EQ(want_state, got_state);
    // Confirm states from `::sample`.
    EXPECT_EQ(want_bitstring, got_bitstring);
    // Confirm states from `::measure`.
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Now place q1 in the 1-state with X-gate.
    qppBackend.x(q1);
    // State vector now `|1> <1| = (0 0 0 1)`.
    want_state = getOneState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("11");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // Confirm state vectors.
    EXPECT_EQ(want_state, got_state);
    // Confirm states from `::sample`.
    EXPECT_EQ(want_bitstring, got_bitstring);
    // Confirm states from `::measure`.
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Send both qubits back to `|0>` with another X-gate.
    qppBackend.x(q0);
    qppBackend.x(q1);
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit Y-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a Y-gate.
    qppBackend.y(q0);
    // State vector for q0 should now be `|q0> = (0 i)`.
    qpp::ket psi_q0 = qpp::ket::Zero(2);
    psi_q0(1) = std::complex<double>(0.0, 1.0);
    // State vector for the system is now `|q0> <0|`.
    want_state = qpp::kron(psi_q0, getZeroState(1));
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Now rotate q1 with a Y-gate.
    qppBackend.y(q1);
    // `|q1> = |q0> = (0 i)`.
    qpp::ket psi_q1 = psi_q0;
    // State vector now `|q0> <q1|`.
    want_state = qpp::kron(psi_q0, psi_q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("11");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Send both qubits back to `|0>` with another Y-gate.
    qppBackend.y(q0);
    qppBackend.y(q1);
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit Z-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a Z-gate.
    qppBackend.z(q0);
    // This should just induce a phase, leaving the state vector
    // unaltered.
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);

    // Now rotate q1 with a Z-gate.
    qppBackend.z(q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // State should still remain unaltered.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit Hadamard-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with an H-gate.
    qppBackend.h(q0);
    // State vector for q0 should now be `|q0> = 1/sqrt(2) * (1 1)`.
    qpp::ket psi_q0 = qpp::ket::Ones(2);
    psi_q0 *= 1 / std::sqrt(2);
    // State vector for the system is now `|q0> <0|`.
    want_state = qpp::kron(psi_q0, getZeroState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    // Now rotate q1 with a H-gate.
    qppBackend.h(q1);
    // `|q1> = |q0> = 1/sqrt(2) * (1 1)`.
    qpp::ket psi_q1 = psi_q0;
    // State vector now `|q0> <q1|`.
    want_state = qpp::kron(psi_q0, psi_q1);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    // Send both qubits back to `|0>` with another H-gate.
    qppBackend.h(q0);
    qppBackend.h(q1);
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit S-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a S-gate.
    qppBackend.s(q0);
    // This should just induce a phase, leaving the state vector
    // unaltered.
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);

    // Now rotate q1 with a S-gate.
    qppBackend.s(q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // State should still remain unaltered.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit T-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a T-gate.
    qppBackend.t(q0);
    // This should just induce a phase, leaving the state vector
    // unaltered.
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);

    // Now rotate q1 with a T-gate.
    qppBackend.t(q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // State should still remain unaltered.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit SDG-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a SDG-gate.
    qppBackend.sdg(q0);
    // This should just induce a phase, leaving the state vector
    // unaltered.
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_state, got_state);

    // Now rotate q1 with a SDG-gate.
    qppBackend.sdg(q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // State should still remain unaltered.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking the single qubit TDG-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Rotate just q0 with a TDG-gate.
    qppBackend.tdg(q0);
    // This should just induce a phase, leaving the state vector
    // unaltered.
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_state, got_state);

    // Now rotate q1 with a TDG-gate.
    qppBackend.tdg(q1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    // State should still remain unaltered.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }
}

// Checking all parameterized 1 and 2 qubit gates.
CUDAQ_TEST(QPPTester, checkParameterizedGates) {
  // Checking RX gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    const int num_qubits = 1;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 by pi with an RX-gate.
    // Note: `RX(pi) = ((0,-i),(-i,0))`
    qppBackend.rx(M_PI, q0);
    // State vector should now be `|psi> = (0 -i)`.
    want_state = -1. * im<> * getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    // Confirm state from `::sample`.
    EXPECT_EQ(want_bitstring, got_bitstring);
    // Confirm state from `::measure`.
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Rotate q0 again by RX(pi).
    // Note: `RX(2*pi) = ((-1,0),(0,-1))` so we should end back
    // at state `(-1, 0)`
    qppBackend.rx(M_PI, q0);
    want_state = -1. * getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
  }

  // Checking RY gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    const int num_qubits = 1;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 by pi with an RY-gate.
    // Note: `RY(pi) = ((0,1),(1,0))` which is equivalent to the
    // conventional Y-gate.
    qppBackend.ry(M_PI, q0);
    // State vector should now be `|psi> = (0 1)`.
    want_state = getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    // Confirm state from `::sample`.
    EXPECT_EQ(want_bitstring, got_bitstring);
    // Confirm state from `::measure`.
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Rotate q0 again by RY(pi).
    // Note: we should end back at `(-1, 0)`
    qppBackend.ry(M_PI, q0);
    want_state = -1. * getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
  }

  // Checking RZ gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    const int num_qubits = 1;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 by 0.0 with an RZ-gate.
    // Should be identity.
    qppBackend.rz(0.0, q0);
    want_state = getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    // Confirm state from `::sample`.
    EXPECT_EQ(want_bitstring, got_bitstring);
    // Confirm state from `::measure`.
    EXPECT_EQ(0, qppBackend.mz(q0));

    // Rotate q0 again by RZ(pi).
    // Note: we should end at `(exp(-i*pi/2)  0)`
    qppBackend.rz(M_PI, q0);
    want_state = std::exp(-0.5 * im<> * M_PI) * getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
  }

  // Checking U gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 with U gate at (theta,phi,lam)=(2pi, 0,0).
    // Note: `U(pi,0,0) = ((-1,0),(0,-1))`
    qppBackend.u3(2 * M_PI, 0.0, 0.0, q0);
    want_state = -1. * getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));

    // Reset q0 to 0 state.
    qppBackend.deallocate(q0);
    q0 = qppBackend.allocateQubit();

    // Rotate the qubit to the 1-state, then rotate it
    // again with another u gate at the same params.
    qppBackend.x(q0);
    qppBackend.u3(2 * M_PI, 0.0, 0.0, q0);
    // State vector should now be: (0, -1)
    want_state = -1. * getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
  }

  // Checking U1 gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 about Z with u1 gate.
    // Note: `U1(pi) = ((1,0),(0,exp(i*pi)))`
    qppBackend.u1(M_PI, q0);
    // State vector shouldn't be affected.
    want_state = getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));

    // Rotate the qubit to the 1-state, then rotate it
    // about Z again with another u1 gate.
    qppBackend.x(q0);
    qppBackend.u1(M_PI, q0);
    // State vector should now be `(0 exp(i*pi))`
    want_state = getOneState(1);
    want_state(1) = std::exp(im<> * M_PI);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
  }

  // Checking U3 gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Rotate q0 with U3 gate at (theta,phi,lam)=(2pi, 0,0).
    // Note: `U3(pi,0,0) = ((-1,0),(0,-1))`
    qppBackend.u3(2 * M_PI, 0.0, 0.0, q0);
    want_state = -1. * getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
    q0 = qppBackend.allocateQubit();

    // Reset system to all 0 state.
    // qppBackend.reset();
    // Rotate the qubit to the 1-state, then rotate it
    // again with another u3 gate at the same params.
    qppBackend.x(q0);
    qppBackend.u3(2 * M_PI, 0.0, 0.0, q0);
    // State vector should now be: (0, -1)
    want_state = -1. * getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    // Confirm state vectors.
    EXPECT_EQ_KETS(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    qppBackend.deallocate(q0);
  }

  // Checking CPHASE gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Starting in state = `|0><0| = |00> = (1 0 0 0)`
    // Apply CPHASE between q0 and q1 at `theta=pi`
    qppBackend.r1(M_PI, /* ctrls */ {q0}, q1);

    // CPHASE shouldn't affect `|00>`
    want_state = getZeroState(2);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    // check that the bitstring from `::sample` is still "00".
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Flip q1 to the 1-state with a regular X-gate.
    // Total system state is now `|01>`.
    qppBackend.x(q1);
    // Apply CPHASE between q0 and q1 again at `theta=pi`.
    qppBackend.r1(M_PI, /* ctrls */ {q0}, q1);

    // CPHASE shouldn't affect `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q0 to the 1 state and q1 back to the 0-state.
    // Total system state is now `|10>`.
    qppBackend.x(q0);
    qppBackend.x(q1);
    // Apply CPHASE between q0 and q1 at `theta=pi`.
    qppBackend.r1(M_PI, /* ctrls */ {q0}, q1);

    // CPHASE shouldn't affect `|10>`
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    // Flip q1 to the 1-state with an X-gate.
    // Total system state is now `|11>`.
    qppBackend.x(q1);
    // Apply CPHASE between q0 and q1 at `theta=pi`
    qppBackend.r1(M_PI, /* ctrls */ {q0}, q1);

    // CS should take `|11>` state vector to (0,0,0,exp(i*theta))
    want_state = std::exp(im<> * M_PI) * getOneState(2);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }
}

// Checking controlled gates called via single qubit gates.
CUDAQ_TEST(QPPTester, checkCtrlGates) {

  // Checking two-qubit controlled X-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Starting in state = `|0><0| = |00> = (1 0 0 0)`
    // Apply controlled-X between q0 and q1
    qppBackend.x(/* ctrls */ {q0}, /* target */ q1);
    // controlled-X shouldn't affect `|00>`
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    // check that the bitstring from `::sample` is still "00".
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Flip q1 to the 1-state with a regular X-gate.
    // Total system state is now `|01>`.
    qppBackend.x(q1);
    // Apply controlled-X between q0 and q1
    qppBackend.x(/* ctrls */ {q0}, /* target */ q1);
    // controlled-X shouldn't affect `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q0 to the 1-state with an X-gate.
    // Total system state is now `|11>`.
    qppBackend.x(q0);
    // Apply controlled-X between q0 and q1
    qppBackend.x(/* ctrls */ {q0}, /* target */ q1);
    // controlled-X should take `|11>` -> `|10>`
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Total system state is still `|10>`.
    // Apply controlled-X between q0 and q1
    qppBackend.x(/* ctrls */ {q0}, /* target */ q1);
    // controlled-X should take `|10>` -> `|11>`
    want_state = qpp::kron(getOneState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("11");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking two-qubit controlled Z-gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Starting in state = `|0><0| = |00> = (1 0 0 0)`
    // Apply controlled-Z between q0 and q1
    qppBackend.z(/* ctrls */ {q0}, /* target */ q1);
    // controlled-Z shouldn't affect `|00>`
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    // check that the bitstring from `::sample` is still "00".
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Flip q1 to the 1-state with a regular X-gate.
    // Total system state is now `|01>`.
    qppBackend.x(q1);
    // Apply controlled-Z between q0 and q1
    qppBackend.z(/* ctrls */ {q0}, /* target */ q1);
    // controlled-X shouldn't affect `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q0 to the 1-state with an X-gate.
    // Total system state is now `|11>`.
    qppBackend.x(q0);
    // Apply controlled-Z between q0 and q1
    qppBackend.z(/* ctrls */ {q0}, /* target */ q1);
    // controlled-Z should take `|11>` state vector to (0,0,0,-1)
    want_state = -1. * qpp::kron(getOneState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("11");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q1 back to the 0-state with an X-gate.
    // Total system state is now `|10>`.
    qppBackend.x(q1);
    // Apply controlled-Z between q0 and q1
    qppBackend.z(/* ctrls */ {q0}, /* target */ q1);
    // controlled-Z shouldn't affect `|10>`
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    // EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking CH gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Starting in state = `|0><0| = |00> = (1 0 0 0)`
    // Apply CH between q0 and q1
    qppBackend.h(/* ctrls */ {q0}, /* target */ q1);
    // CH shouldn't affect `|00>`
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    // check that the bitstring from `::sample` is still "00".
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Flip q1 to the 1-state with a regular X-gate.
    // Total system state is now `|01>`.
    qppBackend.x(q1);
    // Apply CH between q0 and q1
    qppBackend.h(/* ctrls */ {q0}, /* target */ q1);
    // CH shouldn't affect `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q0 to the 1-state with an X-gate.
    // Total system state is now `|11>`.
    qppBackend.x(q0);
    // Apply CH between q0 and q1
    qppBackend.h(/* ctrls */ {q0}, /* target */ q1);
    // CH should take `|11>` state vector to (0,0,1/sqrt(2),-1/sqrt(2))
    want_state = (-1. / sqrt(2)) * getOneState(2);
    want_state(2) = 1.0 / sqrt(2);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    // Reset system to `|00>`.
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
    q0 = qppBackend.allocateQubit();
    q1 = qppBackend.allocateQubit();

    // Flip q0 to the 1 state. Total system state is now `|10>`.
    qppBackend.x(q0);
    // Apply CH between q0 and q1
    qppBackend.h(/* ctrls */ {q0}, /* target */ q1);
    // CH should take `|10>` state vector to (0,0,1/sqrt(2),1/sqrt(2))
    want_state = (1. / sqrt(2)) * getOneState(2);
    want_state(2) = 1.0 / sqrt(2);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking CS gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    const int num_qubits = 2;
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Starting in state = `|0><0| = |00> = (1 0 0 0)`
    // Apply CS between q0 and q1
    qppBackend.s(/* ctrls */ {q0}, /* target */ q1);
    // CS shouldn't affect `|00>`
    want_state = getZeroState(num_qubits);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    // check that the bitstring from `::sample` is still "00".
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Flip q1 to the 1-state with a regular X-gate.
    // Total system state is now `|01>`.
    qppBackend.x(q1);
    // Apply CS between q0 and q1
    qppBackend.s(/* ctrls */ {q0}, /* target */ q1);
    // CS shouldn't affect `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));

    // Flip q0 to the 1 state and q1 back to the 0-state.
    // Total system state is now `|10>`.
    qppBackend.x(q0);
    qppBackend.x(q1);
    // Apply CS between q0 and q1
    qppBackend.s(/* ctrls */ {q0}, /* target */ q1);
    // CH shouldn't affect `|10>`
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);

    // Flip q1 to the 1-state with an X-gate.
    // Total system state is now `|11>`.
    qppBackend.x(q1);
    // Apply CS between q0 and q1
    qppBackend.s(/* ctrls */ {q0}, /* target */ q1);
    // CS should take `|11>` state vector to (0,0,0,i)
    want_state = im<> * getOneState(2);
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }

  // Checking SWAP gate.
  {
    // Keeping track of the state vector.
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Flip q0 to the 1 state.
    qppBackend.x(q0);
    // Ensure the state is now `|10>`
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Swap the qubit states with SWAP gate.
    qppBackend.swap(/* source */ q0, /* target */ q1);
    // Ensure the state is now `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    EXPECT_EQ(want_state, got_state);
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(got_bitstring, want_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));
    qppBackend.deallocate(q0);
    qppBackend.deallocate(q1);
  }
}

CUDAQ_TEST(QPPTester, checkReset) {

  // Testing `::reset()` on 1 qubit.
  {
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Place q0 in the 1 state
    qppBackend.x(q0);
    want_state = getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Reset q0 back to the 0 state
    qppBackend.resetQubit(q0);
    want_state = getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
  }

  // Testing `::reset()` on 2 qubits.
  {
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Place q0 in the 1 state
    qppBackend.x(q0);
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    print_state(got_state);
    std::cout << got_bitstring << "\n";
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Reset system back to the 0 state
    qppBackend.resetQubit(q0);
    want_state = getZeroState(2);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("00");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));
  }

  // Testing `::reset(Qubit)` on 1 qubit.
  {
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 1 qubit
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();

    // Place q0 in the 1 state
    qppBackend.x(q0);
    want_state = getOneState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("1");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));

    // Reset q0 back to the 0 state
    qppBackend.resetQubit(q0);
    want_state = getZeroState(1);
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("0");
    got_bitstring = getSampledBitString(qppBackend, {0});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
  }

  // Testing `::reset(Qubit)` on 2 qubits.
  {
    qpp::ket got_state;
    qpp::ket want_state;
    std::string got_bitstring;
    std::string want_bitstring;
    // Initialize QPP Backend with 2 qubits
    QppCircuitSimulator<qpp::ket> qppBackend;
    auto q0 = qppBackend.allocateQubit();
    auto q1 = qppBackend.allocateQubit();

    // Place q0 in the 1 state
    qppBackend.x(q0);
    want_state = qpp::kron(getOneState(1), getZeroState(1));
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("10");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    std::cout << got_bitstring << "\n";
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(1, qppBackend.mz(q0));
    EXPECT_EQ(0, qppBackend.mz(q1));

    // Reset q0 back to the 0 state and flip q1 to 1 state.
    qppBackend.x(q1);
    qppBackend.resetQubit(q0);
    // State should've flipped to `|01>`
    want_state = qpp::kron(getZeroState(1), getOneState(1));
    got_state = qppBackend.getStateVector();
    want_bitstring = std::string("01");
    got_bitstring = getSampledBitString(qppBackend, {0, 1});
    EXPECT_EQ(want_state, got_state);
    EXPECT_EQ(want_bitstring, got_bitstring);
    EXPECT_EQ(0, qppBackend.mz(q0));
    EXPECT_EQ(1, qppBackend.mz(q1));
  }
}
