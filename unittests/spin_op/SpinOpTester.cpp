/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "utils.h"
#include <gtest/gtest.h>

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

static cudaq::spin_op_term generate_cudaq_spin(int64_t id, int64_t num_qubits,
                                               bool addI = true) {
  constexpr int64_t mask = 0x3;
  auto result = cudaq::spin_op::identity();
  for (int64_t i = 0; i < num_qubits; ++i) {
    switch (paulis[id & mask]) {
    case Pauli::I:
      if (addI)
        result *= cudaq::spin_op::i(i);
      break;
    case Pauli::X:
      result *= cudaq::spin_op::x(i);
      break;
    case Pauli::Y:
      result *= cudaq::spin_op::y(i);
      break;
    case Pauli::Z:
      result *= cudaq::spin_op::z(i);
      break;
    }
    id >>= 2;
  }
  return result;
}

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

TEST(SpinOpTester, checkConstruction) {
  cudaq::spin_op op = cudaq::spin_op::x(10);
  EXPECT_EQ(1, op.num_qubits());
  EXPECT_EQ(1, op.num_terms());
}

TEST(SpinOpTester, checkEquality) {
  auto xx = cudaq::spin_op::x(5);
  EXPECT_EQ(xx, xx);
}

TEST(SpinOpTester, checkFromWord) {
  {
    auto s = cudaq::spin_op::from_word("ZZZ");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(
        cudaq::spin_op::z(0) * cudaq::spin_op::z(1) * cudaq::spin_op::z(2), s);
  }
  {
    auto s = cudaq::spin_op::from_word("XYX");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(
        cudaq::spin_op::x(0) * cudaq::spin_op::y(1) * cudaq::spin_op::x(2), s);
  }
  {
    auto s = cudaq::spin_op::from_word("IZY");
    std::cout << s.to_string() << "\n";
    EXPECT_EQ(
        cudaq::spin_op::i(0) * cudaq::spin_op::z(1) * cudaq::spin_op::y(2), s);
  }
}

TEST(SpinOpTester, checkAddition) {
  cudaq::spin_op op = cudaq::spin_op::x(10);

  auto added = op + op;
  EXPECT_EQ(1, added.num_qubits());
  EXPECT_EQ(1, added.num_terms());
  EXPECT_EQ(2.0, added.begin()->get_coefficient());

  auto added2 =
      cudaq::spin_op::x(0) + cudaq::spin_op::y(1) + cudaq::spin_op::z(2);
  EXPECT_EQ(3, added2.num_terms());
  EXPECT_EQ(3, added2.num_qubits());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(1.0, added2.begin()->get_coefficient());
  }

  auto subtracted = cudaq::spin_op::x(0) - cudaq::spin_op::y(2);
}

TEST(SpinOpTester, checkBug178) {

  cudaq::spin_op op = 1.0 + 2.0 * cudaq::spin_op::x(0);
  auto [bsf, coeffs] = op.get_raw_data();

  std::vector<std::vector<bool>> expected{std::vector<bool>(2),
                                          std::vector<bool>{1, 0}};
  cudaq::spin_op exp(expected, {1., 2.});
  EXPECT_EQ(op, exp);
  EXPECT_EQ(2, op.num_terms());
  EXPECT_EQ(2, exp.num_terms());
  auto op_it = op.begin();
  auto exp_it = exp.begin();
  EXPECT_EQ(*op_it++, *exp_it++);
  EXPECT_EQ(*op_it++, *exp_it++);
  EXPECT_EQ(op_it, op.end());
  EXPECT_EQ(exp_it, exp.end());
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
        auto a_spin = generate_cudaq_spin(i, num_qubits);
        auto b_spin = generate_cudaq_spin(j, num_qubits, false);
        auto result_spin = a_spin * b_spin;

        // Check result
        EXPECT_EQ(generate_pauli_string(result), result_spin.to_string(false));
        EXPECT_EQ(phase, result_spin.get_coefficient());
      }
    }
  }
}

TEST(SpinOpTester, canBuildDeuteron) {
  auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
           2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
           .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  EXPECT_EQ(5, H.num_terms());
  EXPECT_EQ(2, H.num_qubits());
}

TEST(SpinOpTester, checkGetSparseMatrix) {
  auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
           2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
           .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

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
  auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
           2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
           .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  auto matrix = H.to_matrix();
  matrix.dump();
  auto groundEnergy = matrix.minimal_eigenvalue();

  {
    EXPECT_NEAR(groundEnergy.real(), -1.74, 1e-2);

    std::vector<double> expected{.00029,  0, 0, 0,       0,       -.43619,
                                 -4.2866, 0, 0, -4.2866, 12.2503, 0,
                                 0,       0, 0, 11.8137};
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j)
        EXPECT_NEAR(matrix(i, j).real(), expected[4 * i + j], 1e-3);
    }
  }
  {
    // Create the G=ground state for the above hamiltonian
    cudaq::complex_matrix vec(4, 1);
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
    EXPECT_EQ(tmp.size(), vec.size());

    // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
    for (std::size_t i = 0; i < 4; i++)
      EXPECT_NEAR(tmp[i].real(), groundEnergy.real() * vec[i].real(), 1e-2);
  }

  {
    // test <psi_g | H | psi_g>
    cudaq::complex_matrix psig(4, 1);
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
  auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
           2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
           .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  std::size_t count = 0;
  for (auto term : H) {
    std::cout << "TEST: " << term.to_string();
    count++;
  }

  EXPECT_EQ(count, H.num_terms());
}

TEST(SpinOpTester, checkDistributeTerms) {
  auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
           2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
           .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  auto distributed = H.distribute_terms(2);

  EXPECT_EQ(distributed.size(), 2);
  EXPECT_EQ(distributed[0].num_terms(), 3);
  EXPECT_EQ(distributed[1].num_terms(), 2);
}

TEST(SpinOpTester, checkMultiDiagConversionSpin) {
  for (auto &H : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                  cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
    utils::checkEqual(H.to_matrix(), H.to_diagonal_matrix());
  }

  // Product ops testing
  for (auto &H1 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                   cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
    for (auto &H2 : {cudaq::spin_op::i(1), cudaq::spin_op::x(1),
                     cudaq::spin_op::y(1), cudaq::spin_op::z(1)}) {
      auto H = H1 * H2;
      std::cout << "Testing " << H.to_string() << "\n";
      utils::checkEqual(H.to_matrix(), H.to_diagonal_matrix());
    }
  }

  for (auto &H1 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                   cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
    for (auto &H2 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                     cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
      auto H = H1 * H2;
      std::cout << "Testing " << H.to_string() << "\n";
      utils::checkEqual(H.to_matrix(), H.to_diagonal_matrix());
    }
  }

  // Sum ops testing
  for (auto &H1 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                   cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
    for (auto &H2 : {cudaq::spin_op::i(1), cudaq::spin_op::x(1),
                     cudaq::spin_op::y(1), cudaq::spin_op::z(1)}) {
      for (auto &H3 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                       cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
        for (auto &H4 : {cudaq::spin_op::i(1), cudaq::spin_op::x(1),
                         cudaq::spin_op::y(1), cudaq::spin_op::z(1)}) {
          auto H = H1 * H2 + H3 * H4;
          std::cout << "Testing " << H.to_string() << "\n";
          utils::checkEqual(H.to_matrix(), H.to_diagonal_matrix());
        }
      }
    }
  }

  for (auto &H1 : {cudaq::spin_op::i(0), cudaq::spin_op::x(0),
                   cudaq::spin_op::y(0), cudaq::spin_op::z(0)}) {
    for (auto &H2 : {cudaq::spin_op::i(1), cudaq::spin_op::x(1),
                     cudaq::spin_op::y(1), cudaq::spin_op::z(1)}) {
      for (auto &H3 : {cudaq::spin_op::i(1), cudaq::spin_op::x(1),
                       cudaq::spin_op::y(1), cudaq::spin_op::z(1)}) {
        for (auto &H4 : {cudaq::spin_op::i(2), cudaq::spin_op::x(2),
                         cudaq::spin_op::y(2), cudaq::spin_op::z(2)}) {
          auto H = H1 * H2 + H3 * H4;
          std::cout << "Testing " << H.to_string() << "\n";
          utils::checkEqual(H.to_matrix(), H.to_diagonal_matrix());
        }
      }
    }
  }
}

#if (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))
#pragma GCC diagnostic pop
#endif
#ifdef __clang__
#pragma clang diagnostic pop
#endif
