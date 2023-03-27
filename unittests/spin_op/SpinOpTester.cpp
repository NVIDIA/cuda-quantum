/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq/spin_op.h"

using namespace cudaq::spin;

TEST(SpinOpTester, checkConstruction) {
  cudaq::spin_op op = x(10);
  EXPECT_EQ(11, op.n_qubits());
  EXPECT_EQ(1, op.n_terms());
}

TEST(SpinOpTester, checkEquality) {
  cudaq::spin_op xx = x(5);
  EXPECT_EQ(xx, xx);
}

TEST(SpinOpTester, checkAddition) {
  cudaq::spin_op op = x(10);

  auto added = op + op;
  EXPECT_EQ(11, added.n_qubits());
  EXPECT_EQ(1, added.n_terms());
  EXPECT_EQ(2.0, added.get_term_coefficient(0));

  op.dump();
  added.dump();

  auto added2 = x(0) + y(1) + z(2);
  added2.dump();
  EXPECT_EQ(3, added2.n_terms());
  EXPECT_EQ(3, added2.n_qubits());
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(1.0, added2.get_term_coefficient(i));
  }

  auto subtracted = x(0) - y(2);
  subtracted.dump();
}

TEST(SpinOpTester, checkBug178) {

  cudaq::spin_op op = 1.0 + 2.0 * x(0);
  op.dump();
  auto bsf = op.get_bsf();

  std::vector<std::vector<bool>> expected{std::vector<bool>(2),
                                          std::vector<bool>{1, 0}};

  for (std::size_t counter = 0; auto &row : bsf) {
    EXPECT_EQ(expected[counter++], row);
  }
}

TEST(SpinOpTester, checkMultiplication) {
  auto mult = x(0) * y(1);
  mult.dump();

  auto tmp = 2 * y(1);
  tmp.dump();

  auto mult2 = x(3) * tmp;
  mult2.dump();

  std::cout << "X * Z: -iY\n";
  (x(3) * z(3)).dump();
  EXPECT_EQ(y(3), x(3) * z(3));
  EXPECT_EQ((x(3) * z(3)).get_term_coefficient(0), std::complex<double>(0, -1));

  std::cout << "X * X: I\n";
  (x(2) * x(2)).dump();
  EXPECT_EQ(cudaq::spin_op(), x(2) * x(2));
  EXPECT_EQ((x(2) * x(2)).get_term_coefficient(0), std::complex<double>(1, 0));

  std::cout << "Y * Y: I\n";
  (y(14) * y(14)).dump();
  EXPECT_EQ(cudaq::spin_op(), y(14) * y(14));
  EXPECT_EQ((y(14) * y(14)).get_term_coefficient(0),
            std::complex<double>(1, 0));

  std::cout << "Z * Z: I\n";
  (z(0) * z(0)).dump();
  EXPECT_EQ(cudaq::spin_op(), z(0) * z(0));
  EXPECT_EQ((z(0) * z(0)).get_term_coefficient(0), std::complex<double>(1, 0));

  std::cout << "X * Y: iZ\n";
  (x(3) * y(3)).dump();
  EXPECT_EQ(z(3), x(3) * y(3));
  EXPECT_EQ((x(3) * y(3)).get_term_coefficient(0), std::complex<double>(0, 1));

  std::cout << "I * I: I\n";
  (i(2) * i(2)).dump();
  EXPECT_EQ(i(2), i(2) * i(2));
  EXPECT_EQ((i(2) * i(2)).get_term_coefficient(0), std::complex<double>(1, 0));

  std::cout << "I * Z: Z\n";
  (i(3) * i(3)).dump();
  EXPECT_EQ(z(3), i(3) * z(3));
  EXPECT_EQ((i(3) * z(3)).get_term_coefficient(0), std::complex<double>(1, 0));
}

TEST(SpinOpTester, canBuildDeuteron) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);

  H.dump();

  EXPECT_EQ(5, H.n_terms());
  EXPECT_EQ(2, H.n_qubits());

  for (std::size_t i = 0; i < H.n_terms(); i++) {
    H[i].dump();
  }
}

TEST(SpinOpTester, checkGetMatrix) {
  auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + .21829 * z(0) -
           6.125 * z(1);
  auto matrix = H.to_matrix();
  matrix.dump();
  auto groundEnergy = matrix.minimal_eigenvalue();
  EXPECT_NEAR(groundEnergy.real(), -1.74, 1e-2);

  std::vector<double> expected{.00029,  0, 0, 0,       0,       12.2503,
                               -4.2866, 0, 0, -4.2866, -.43619, 0,
                               0,       0, 0, 11.8137};
  for (std::size_t i = 0; i < 16; ++i)
    EXPECT_NEAR(matrix.data()[i].real(), expected[i], 1e-3);

  // Create the G=ground state for the above hamiltonian
  cudaq::complex_matrix vec(4, 1);
  vec.set_zero();
  vec(1, 0) = .292786;
  vec(2, 0) = .956178;

  // Compute H |psi_g>
  auto tmp = matrix * vec;
 
  // Should have H |psi_g> = E |psi_g> (eigenvalue equation)
  for (std::size_t i = 0; i < 4; i++)
    EXPECT_NEAR(tmp(i, 0).real(), groundEnergy.real() * vec(i, 0).real(), 1e-2);
}