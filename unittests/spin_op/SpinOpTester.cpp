/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/spin_op.h"

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
  auto mult = x(0) * y(1);
  mult.dump();
  auto mult3 = y(0) * y(1);
  mult3.dump();

  auto tmp = 2 * y(1);
  tmp.dump();

  auto mult2 = x(3) * tmp;
  mult2.dump();

  std::cout << "X * Y: iZ\n";
  (x(3) * y(3)).dump();
  EXPECT_EQ(z(3), x(3) * y(3));
  EXPECT_EQ((x(3) * y(3)).get_coefficient(), std::complex<double>(0, 1));

  std::cout << "Y * Z: iX\n";
  (y(3) * z(3)).dump();
  EXPECT_EQ(x(3), y(3) * z(3));
  EXPECT_EQ((y(3) * z(3)).get_coefficient(), std::complex<double>(0, 1));

  std::cout << "Z * X: iY\n";
  (z(3) * x(3)).dump();
  EXPECT_EQ(y(3), z(3) * x(3));
  EXPECT_EQ((z(3) * x(3)).get_coefficient(), std::complex<double>(0, 1));

  std::cout << "Y * X: -iZ\n";
  (y(3) * x(3)).dump();
  EXPECT_EQ(z(3), y(3) * x(3));
  EXPECT_EQ((y(3) * x(3)).get_coefficient(), std::complex<double>(0, -1));

  std::cout << "Z * Y: -iX\n";
  (z(3) * y(3)).dump();
  EXPECT_EQ(x(3), z(3) * y(3));
  EXPECT_EQ((z(3) * y(3)).get_coefficient(), std::complex<double>(0, -1));

  std::cout << "X * Z: -iY\n";
  (x(3) * z(3)).dump();
  EXPECT_EQ(y(3), x(3) * z(3));
  EXPECT_EQ((x(3) * z(3)).get_coefficient(), std::complex<double>(0, -1));

  std::cout << "X * X: I\n";
  (x(2) * x(2)).dump();
  EXPECT_EQ(cudaq::spin_op(), x(2) * x(2));
  EXPECT_EQ((x(2) * x(2)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "Y * Y: I\n";
  (y(14) * y(14)).dump();
  EXPECT_EQ(cudaq::spin_op(), y(14) * y(14));
  EXPECT_EQ((y(14) * y(14)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "Z * Z: I\n";
  (z(0) * z(0)).dump();
  EXPECT_EQ(cudaq::spin_op(), z(0) * z(0));
  EXPECT_EQ((z(0) * z(0)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "I * I: I\n";
  (i(2) * i(2)).dump();
  EXPECT_EQ(i(2), i(2) * i(2));
  EXPECT_EQ((i(2) * i(2)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "I * X: X\n";
  (i(3) * i(3)).dump();
  EXPECT_EQ(x(3), i(3) * x(3));
  EXPECT_EQ((i(3) * x(3)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "I * Y: Y\n";
  (i(3) * i(3)).dump();
  EXPECT_EQ(y(3), i(3) * y(3));
  EXPECT_EQ((i(3) * y(3)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "I * Z: Z\n";
  (i(3) * i(3)).dump();
  EXPECT_EQ(z(3), i(3) * z(3));
  EXPECT_EQ((i(3) * z(3)).get_coefficient(), std::complex<double>(1, 0));

  std::cout << "Eliminate zero coefficient terms\n";
  auto result = (0.0 * x(0)) * z(1);
  result.dump();
  EXPECT_EQ(result.num_terms(), 0);

  std::cout << "X(0) * Y(0) = -Y(0) * X(0)\n";
  auto left_side = x(0) * y(0);
  auto right_side = -1.0 * y(0) * x(0);
  left_side.dump();
  right_side.dump();
  EXPECT_EQ(left_side, right_side);

  std::cout << "X(0) * (Z(0) * Z(1)) = (Z(0) * Z(1)) * X(0)\n";
  left_side = x(0) * (z(0) + z(1));
  right_side = (z(0) + z(1)) * x(0);
  left_side.dump();
  right_side.dump();
  EXPECT_EQ(left_side, right_side);

  std::cout << "(X(0) * Y(1)) * Z(2) = X(0) * (Y(1) * Z(2))\n";
  left_side = (x(0) * y(1)) * z(2);
  right_side = x(0) * (y(1) * z(2));
  left_side.dump();
  right_side.dump();
  EXPECT_EQ(left_side, right_side);

  std::cout << "X(0) * (Z(0) * Z(1) + Y(0) * Y(1))\n";
  result = x(0) * (z(0) * z(1) + y(0) * y(1));
  result.dump();
  auto expected = x(0) * z(0) * z(1) + x(0) * y(0) * y(1);
  EXPECT_EQ(result, expected);

  auto tmp2 = 2 * x(0) * x(1) * y(2) * y(3) + 3 * y(0) * y(1) * x(2) * x(3);
  std::cout << "START\n";
  tmp2 = tmp2 * tmp2;
  tmp2.dump();

  EXPECT_EQ(2, tmp2.num_terms());
  expected = 13 * i(0) * i(1) * i(2) * i(3) + 12 * z(0) * z(1) * z(2) * z(3);
  EXPECT_EQ(expected, tmp2);
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
