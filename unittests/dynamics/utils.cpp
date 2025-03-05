/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/utils/matrix.h"
#include <gtest/gtest.h>
#include <iostream>

namespace utils {

void print(cudaq::complex_matrix mat, std::string name = "") {
  if (name != "")
    std::cout << name << ":" << std::endl;
  for (std::size_t i = 0; i < mat.rows(); i++) {
    for (std::size_t j = 0; j < mat.cols(); j++)
      std::cout << mat[{i, j}] << " ";
    std::cout << std::endl;
  }
}

void assert_product_equal(
    const cudaq::product_operator<cudaq::matrix_operator> &got,
    const std::complex<double> &expected_coefficient,
    const std::vector<cudaq::matrix_operator> &expected_terms) {
  cudaq::operator_sum<cudaq::matrix_operator> sum = got;
  ASSERT_TRUE(sum.get_terms().size() == 1);
  ASSERT_TRUE(got.get_coefficient().evaluate() == expected_coefficient);
  ASSERT_TRUE(got.get_terms() == expected_terms);
}

void checkEqual(cudaq::complex_matrix a, cudaq::complex_matrix b) {
  print(a, "matrix a");
  print(b, "matrix b");

  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.rows(), b.rows());
  ASSERT_EQ(a.cols(), b.cols());
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.rows(); i++) {
    for (std::size_t j = 0; j < a.cols(); j++) {
      auto a_val = a[{i, j}];
      auto b_val = b[{i, j}];
      EXPECT_NEAR(a_val.real(), b_val.real(), 1e-8);
      EXPECT_NEAR(a_val.imag(), b_val.imag(), 1e-8);
    }
  }
}

cudaq::complex_matrix zero_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  return mat;
}

cudaq::complex_matrix id_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = 1.0;
  return mat;
}

cudaq::complex_matrix annihilate_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
  return mat;
}

cudaq::complex_matrix create_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
  return mat;
}

cudaq::complex_matrix position_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] = 0.5 * std::sqrt(static_cast<double>(i + 1));
    mat[{i, i + 1}] = 0.5 * std::sqrt(static_cast<double>(i + 1));
  }
  return mat;
}

cudaq::complex_matrix momentum_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] =
        std::complex<double>(0., 0.5) * std::sqrt(static_cast<double>(i + 1));
    mat[{i, i + 1}] =
        std::complex<double>(0., -0.5) * std::sqrt(static_cast<double>(i + 1));
  }
  return mat;
}

cudaq::complex_matrix number_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = static_cast<double>(i);
  return mat;
}

cudaq::complex_matrix parity_matrix(std::size_t size) {
  auto mat = cudaq::complex_matrix(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = std::pow(-1., static_cast<double>(i));
  return mat;
}

cudaq::complex_matrix displace_matrix(std::size_t size,
                                std::complex<double> amplitude) {
  auto term1 = amplitude * create_matrix(size);
  auto term2 = std::conj(amplitude) * annihilate_matrix(size);
  auto difference = term1 - term2;
  return difference.exponential();
}

cudaq::complex_matrix squeeze_matrix(std::size_t size,
                               std::complex<double> amplitude) {
  auto term1 = std::conj(amplitude) * annihilate_matrix(size).power(2);
  auto term2 = amplitude * create_matrix(size).power(2);
  auto difference = 0.5 * (term1 - term2);
  return difference.exponential();
}

cudaq::complex_matrix PauliX_matrix() {
  auto mat = cudaq::complex_matrix(2, 2);
  mat[{0, 1}] = 1.0;
  mat[{1, 0}] = 1.0;
  return mat;
}

cudaq::complex_matrix PauliZ_matrix() {
  auto mat = cudaq::complex_matrix(2, 2);
  mat[{0, 0}] = 1.0;
  mat[{1, 1}] = -1.0;
  return mat;
}

cudaq::complex_matrix PauliY_matrix() {
  return std::complex<double>(0., 1.) * utils::PauliX_matrix() *
         utils::PauliZ_matrix();
}

} // namespace utils
