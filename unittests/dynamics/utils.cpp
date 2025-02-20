/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/utils/tensor.h"
#include <gtest/gtest.h>
#include <iostream>

namespace utils {

void print(cudaq::matrix_2 mat, std::string name = "") {
  if (name != "")
    std::cout << name << ":" << std::endl;
  for (std::size_t i = 0; i < mat.get_rows(); i++) {
    for (std::size_t j = 0; j < mat.get_columns(); j++)
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

void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b) {
  print(a, "matrix a");
  print(b, "matrix b");

  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      auto a_val = a[{i, j}];
      auto b_val = b[{i, j}];
      EXPECT_NEAR(a_val.real(), b_val.real(), 1e-8);
      EXPECT_NEAR(a_val.imag(), b_val.imag(), 1e-8);
    }
  }
}

cudaq::matrix_2 zero_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  return mat;
}

cudaq::matrix_2 id_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = 1.0 + 0.0j;
  return mat;
}

cudaq::matrix_2 annihilate_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 create_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++)
    mat[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  return mat;
}

cudaq::matrix_2 position_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] = 0.5 * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 momentum_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i + 1 < size; i++) {
    mat[{i + 1, i}] =
        (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
    mat[{i, i + 1}] =
        (-0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
  }
  return mat;
}

cudaq::matrix_2 number_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = static_cast<double>(i) + 0.0j;
  return mat;
}

cudaq::matrix_2 parity_matrix(std::size_t size) {
  auto mat = cudaq::matrix_2(size, size);
  for (std::size_t i = 0; i < size; i++)
    mat[{i, i}] = std::pow(-1., static_cast<double>(i)) + 0.0j;
  return mat;
}

cudaq::matrix_2 displace_matrix(std::size_t size,
                                std::complex<double> amplitude) {
  auto term1 = amplitude * create_matrix(size);
  auto term2 = std::conj(amplitude) * annihilate_matrix(size);
  auto difference = term1 - term2;
  return difference.exponential();
}

cudaq::matrix_2 squeeze_matrix(std::size_t size,
                               std::complex<double> amplitude) {
  auto term1 = std::conj(amplitude) * annihilate_matrix(size).power(2);
  auto term2 = amplitude * create_matrix(size).power(2);
  auto difference = 0.5 * (term1 - term2);
  return difference.exponential();
}

cudaq::matrix_2 PauliX_matrix() {
  auto mat = cudaq::matrix_2(2, 2);
  mat[{0, 1}] = 1.0;
  mat[{1, 0}] = 1.0;
  return mat;
}

cudaq::matrix_2 PauliZ_matrix() {
  auto mat = cudaq::matrix_2(2, 2);
  mat[{0, 0}] = 1.0;
  mat[{1, 1}] = -1.0;
  return mat;
}

cudaq::matrix_2 PauliY_matrix() {
  return 1.0j * utils::PauliX_matrix() * utils::PauliZ_matrix();
}

} // namespace utils
