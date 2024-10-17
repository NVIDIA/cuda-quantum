/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/spin_op.h"

enum Pauli : int8_t { I = 0, X, Y, Z };
constexpr Pauli paulis[4] = {Pauli::I, Pauli::X, Pauli::Y, Pauli::Z};

// Function to multiply two single-qubit Pauli operators
static std::pair<std::complex<double>, Pauli> multiply_paulis(Pauli a,
                                                              Pauli b) {
  using namespace std::complex_literals;
  // I    X    Y    Z
  constexpr std::complex<double> table[4][4] = {
      {1., 1., 1, 1},    // I
      {1., 1., 1i, -1i}, // X
      {1., -1i, 1, 1i},  // Y
      {1., 1i, -1i, 1}   // Z
  };
  if (a == b)
    return {1.0, Pauli::I};
  if (a == Pauli::I)
    return {1.0, b};
  if (b == Pauli::I)
    return {1.0, a};
  return {table[a][b], paulis[a ^ b]};
}

// Function to multiply two multi-qubit Pauli words
static std::pair<std::complex<double>, std::vector<Pauli>>
multiply_pauli_words(const std::vector<Pauli> &a, const std::vector<Pauli> &b,
                     bool verbose = false) {
  std::complex<double> phase = 1.0;
  std::string info;
  std::vector<Pauli> result(a.size(), Pauli::I);
  for (size_t i = 0; i < a.size(); ++i) {
    auto [p, r] = multiply_paulis(a[i], b[i]);
    phase *= p;
    result[i] = r;
  }
  return {phase, result};
}

// Generates a pauli word out of a binary representation of it.
static std::vector<Pauli> generate_pauli_word(int64_t id, int64_t num_qubits) {
  constexpr int64_t mask = 0x3;
  std::vector<Pauli> word(num_qubits, Pauli::I);
  for (int64_t i = 0; i < num_qubits; ++i) {
    assert((id & mask) < 4);
    word[i] = paulis[id & mask];
    id >>= 2;
  }
  return word;
}

static std::string generate_pauli_string(const std::vector<Pauli> &word) {
  constexpr char paulis_name[4] = {'I', 'X', 'Y', 'Z'};
  std::string result(word.size(), 'I');
  for (int64_t i = 0; i < word.size(); ++i)
    result[i] = paulis_name[word[i]];
  return result;
}

static cudaq::spin_op generate_cudaq_spin(int64_t id, int64_t num_qubits,
                                          bool addI = true) {
  constexpr int64_t mask = 0x3;
  cudaq::spin_op result;
  for (int64_t i = 0; i < num_qubits; ++i) {
    switch (paulis[id & mask]) {
    case Pauli::I:
      if (addI)
        result *= cudaq::spin::i(i);
      break;
    case Pauli::X:
      result *= cudaq::spin::x(i);
      break;
    case Pauli::Y:
      result *= cudaq::spin::y(i);
      break;
    case Pauli::Z:
      result *= cudaq::spin::z(i);
      break;
    }
    id >>= 2;
  }
  return result;
}

using namespace cudaq::spin;

TEST(SpinOpTester, checkConstruction) {
  cudaq::spin_op op = x(10);
  EXPECT_EQ(11, op.num_qubits());
  EXPECT_EQ(1, op.num_terms());
}

TEST(SpinOpTester, checkEquality) {
  cudaq::spin_op xx = x(5);
  EXPECT_EQ(xx, xx);
}

TEST(SpinOpTester, checkFromWord) {
  {
    auto s = cudaq::spin_op::from_word("ZZZ");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(z(0) * z(1) * z(2), s);
  }
  {
    auto s = cudaq::spin_op::from_word("XYX");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(x(0) * y(1) * x(2), s);
  }
  {
    auto s = cudaq::spin_op::from_word("IZY");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(i(0) * z(1) * y(2), s);
  }
}

TEST(SpinOpTester, checkAddition) {
  cudaq::spin_op op = x(10);

  auto added = op + op;
  EXPECT_EQ(11, added.num_qubits());
  EXPECT_EQ(1, added.num_terms());
  EXPECT_EQ(2.0, added.get_coefficient());

  op.dump();
  added.dump();

  auto added2 = x(0) + y(1) + z(2);
  added2.dump();
  EXPECT_EQ(3, added2.num_terms());
  EXPECT_EQ(3, added2.num_qubits());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(1.0, added2.begin()->get_coefficient());
  }

  auto subtracted = x(0) - y(2);
  subtracted.dump();
}

TEST(SpinOpTester, checkBug178) {

  cudaq::spin_op op = 1.0 + 2.0 * x(0);
  op.dump();
  auto [bsf, coeffs] = op.get_raw_data();

  std::vector<std::vector<bool>> expected{std::vector<bool>(2),
                                          std::vector<bool>{1, 0}};
  cudaq::spin_op exp(expected, {1., 2.});
  EXPECT_EQ(op, exp);

  EXPECT_TRUE(std::find(expected.begin(), expected.end(), bsf[0]) !=
              expected.end());
  EXPECT_TRUE(std::find(expected.begin(), expected.end(), bsf[1]) !=
              expected.end());
}

TEST(SpinOpTester, checkMultiplication) {
  for (int num_qubits = 1; num_qubits <= 4; ++num_qubits) {
    int64_t num_words = std::pow(4, num_qubits);
    for (int64_t i = 0; i < num_words; ++i) {
      for (int64_t j = 0; j < num_words; ++j) {
        // Expected result:
        std::vector<Pauli> a_word = generate_pauli_word(i, num_qubits);
        std::vector<Pauli> b_word = generate_pauli_word(j, num_qubits);
        auto [phase, result] = multiply_pauli_words(a_word, b_word);

        // Result:
        cudaq::spin_op a_spin = generate_cudaq_spin(i, num_qubits);
        cudaq::spin_op b_spin = generate_cudaq_spin(j, num_qubits, false);
        cudaq::spin_op result_spin = a_spin * b_spin;

        // Check result
        EXPECT_EQ(generate_pauli_string(result), result_spin.to_string(false));
        EXPECT_EQ(phase, result_spin.get_coefficient());
      }
    }
  }
}

TEST(SpinOpTester, canBuildDeuteron) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);

  H.dump();

  EXPECT_EQ(5, H.num_terms());
  EXPECT_EQ(2, H.num_qubits());
}

TEST(SpinOpTester, checkGetSparseMatrix) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);
  auto matrix = H.to_matrix();
  matrix.dump();
  auto [values, rows, cols] = H.to_sparse_matrix();
  for (std::size_t i = 0; auto &el : values) {
    std::cout << rows[i] << ", " << cols[i] << ", " << el << "\n";
    EXPECT_NEAR(matrix(rows[i], cols[i]).real(), el.real(), 1e-3);
    i++;
  }
}

TEST(SpinOpTester, checkGetMatrix) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);
  auto matrix = H.to_matrix();
  matrix.dump();
  auto groundEnergy = matrix.minimal_eigenvalue();

  {
    EXPECT_NEAR(groundEnergy.real(), -1.74, 1e-2);

    std::vector<double> expected{.00029,  0, 0, 0,       0,       -.43619,
                                 -4.2866, 0, 0, -4.2866, 12.2503, 0,
                                 0,       0, 0, 11.8137};
    for (std::size_t i = 0; i < 16; ++i)
      EXPECT_NEAR(matrix.data()[i].real(), expected[i], 1e-3);
  }
  {
    // Create the G=ground state for the above hamiltonian
    cudaq::complex_matrix vec(4, 1);
    vec.set_zero();
    vec(2, 0) = .292786;
    vec(1, 0) = .956178;

    // Compute H |psi_g>
    auto tmp = matrix * vec;

    // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
    for (std::size_t i = 0; i < 4; i++)
      EXPECT_NEAR(tmp(i, 0).real(), groundEnergy.real() * vec(i, 0).real(),
                  1e-2);
  }
  {
    // do the same thing, but use std vector instead
    std::vector<std::complex<double>> vec{0.0, .956178, .292786, 0.0};

    // Compute H |psi_g>
    auto tmp = matrix * vec;

    // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
    for (std::size_t i = 0; i < 4; i++)
      EXPECT_NEAR(tmp(i, 0).real(), groundEnergy.real() * vec[i].real(), 1e-2);
  }

  {
    // test <psi_g | H | psi_g>
    cudaq::complex_matrix psig(4, 1);
    psig.set_zero();
    psig(2, 0) = .292786;
    psig(1, 0) = .956178;
    auto prod = matrix * psig;
    double sum = 0.0;
    for (std::size_t i = 0; i < 4; i++) {
      sum += (std::conj(psig(i, 0)) * prod(i, 0)).real();
    }
    EXPECT_NEAR(sum, -1.74, 1e-2);
  }
}

TEST(SpinOpTester, checkIterator) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);

  std::size_t count = 0;
  for (auto term : H) {
    std::cout << "TEST: " << term.to_string();
    count++;
  }

  EXPECT_EQ(count, H.num_terms());
}

TEST(SpinOpTester, checkDistributeTerms) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);

  auto distributed = H.distribute_terms(2);

  EXPECT_EQ(distributed.size(), 2);
  EXPECT_EQ(distributed[0].num_terms(), 3);
  EXPECT_EQ(distributed[1].num_terms(), 2);
}
