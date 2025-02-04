/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include <gtest/gtest.h>

namespace utils_0 {
void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b) {
  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      double a_real = a[{i, j}].real();
      double b_real = b[{i, j}].real();
      EXPECT_NEAR(a_real, b_real, 1e-8);
      double a_imag = a[{i, j}].imag();
      double b_imag = b[{i, j}].imag();
      EXPECT_NEAR(a_imag, b_imag, 1e-8);
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
        -1. * (0.5j) * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
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

void assert_product_equal(
    const cudaq::product_operator<cudaq::matrix_operator> &got,
    const std::complex<double> &expected_coefficient,
    const std::vector<cudaq::matrix_operator> &expected_terms) {

  auto sumterms_prod =
      ((const cudaq::operator_sum<cudaq::matrix_operator> &)got).get_terms();
  ASSERT_TRUE(sumterms_prod.size() == 1);
  ASSERT_TRUE(got.get_coefficient().evaluate() == expected_coefficient);
  ASSERT_TRUE(got.get_terms() == expected_terms);
}

} // namespace utils_0

TEST(OperatorExpressions, checkElementaryUnary) {
  auto id = cudaq::matrix_operator::identity(0);
  utils_0::checkEqual((-id).to_matrix({{0,2}}), -1.0 * utils_0::id_matrix(2));
}

TEST(OperatorExpressions, checkElementaryAgainstDouble) {
  std::complex<double> value = 0.125 + 0.125j;

  // `matrix_operator` + `complex<double>` and `complex<double>` +
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::annihilate(0);

    auto sum = value + elementary;
    auto reverse = elementary + value;

    auto got_matrix = sum.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils_0::id_matrix(3);
    auto want_matrix = scaled_identity + utils_0::annihilate_matrix(3);
    auto want_matrix_reverse = utils_0::annihilate_matrix(3) + scaled_identity;

    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `matrix_operator` - `complex<double>` and `complex<double>` -
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::position(0);

    auto difference = value - elementary;
    auto reverse = elementary - value;

    auto got_matrix = difference.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils_0::id_matrix(3);
    auto want_matrix = scaled_identity - utils_0::position_matrix(3);
    auto want_matrix_reverse = utils_0::position_matrix(3) - scaled_identity;

    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `matrix_operator` * `complex<double>` and `complex<double>` *
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::number(0);

    auto product = value * elementary;
    auto reverse = elementary * value;

    auto got_matrix = product.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils_0::id_matrix(3);
    auto want_matrix = scaled_identity * utils_0::number_matrix(3);
    auto want_matrix_reverse = utils_0::number_matrix(3) * scaled_identity;

    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

TEST(OperatorExpressions, checkPreBuiltElementaryOpsScalars) {

  auto function = [](std::map<std::string, std::complex<double>> parameters) {
    return parameters["value"];
  };

  /// Keeping these fixed for these more simple tests.
  int level_count = 3;
  int degree_index = 0;
  double const_scale_factor = 2.0;

  // `matrix_operator + scalar_operator`
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix =
        utils_0::annihilate_matrix(level_count) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils_0::annihilate_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator + scalar_operator`
  {
    auto self = cudaq::matrix_operator::parity(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}},
                                    {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils_0::parity_matrix(level_count) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils_0::parity_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator - scalar_operator`
  {
    auto self = cudaq::matrix_operator::number(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix = utils_0::number_matrix(level_count) - scaled_identity;
    auto want_reverse_matrix =
        scaled_identity - utils_0::number_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator - scalar_operator`
  {
    auto self = cudaq::matrix_operator::position(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}},
                                    {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils_0::position_matrix(level_count) - scaled_identity;
    auto want_reverse_matrix =
        scaled_identity - utils_0::position_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator * scalar_operator`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto product = self * other;
    auto reverse = other * self;

    utils_0::assert_product_equal(product, const_scale_factor,
                                  {cudaq::matrix_operator("momentum", {0})});
    utils_0::assert_product_equal(reverse, const_scale_factor,
                                  {cudaq::matrix_operator("momentum", {0})});

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = product.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix = utils_0::momentum_matrix(level_count) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils_0::momentum_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator * scalar_operator`
  {
    auto self = cudaq::matrix_operator::create(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    utils_0::assert_product_equal(product, other.evaluate(),
                                  {cudaq::matrix_operator("create", {0})});
    utils_0::assert_product_equal(reverse, other.evaluate(),
                                  {cudaq::matrix_operator("create", {0})});

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils_0::id_matrix(level_count);
    auto got_matrix = product.to_matrix({{degree_index, level_count}},
                                        {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils_0::create_matrix(level_count) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils_0::create_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }
}

/// Prebuilt elementary ops against one another.
TEST(OperatorExpressions, checkPreBuiltElementaryOpsSelf) {

  /// Keeping this fixed throughout.
  int level_count = 3;
  std::map<int, int> dimensions = {{0, level_count}, {1, level_count}};

  // Addition, same DOF.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.n_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils_0::annihilate_matrix(level_count) +
                       utils_0::create_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.n_terms() == 2);

    auto annihilate_full =
        cudaq::kronecker(utils_0::id_matrix(level_count),
                         utils_0::annihilate_matrix(level_count));
    auto create_full = cudaq::kronecker(utils_0::create_matrix(level_count),
                                        utils_0::id_matrix(level_count));
    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = annihilate_full + create_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.n_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils_0::annihilate_matrix(level_count) -
                       utils_0::create_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.n_terms() == 2);

    auto annihilate_full =
        cudaq::kronecker(utils_0::id_matrix(level_count),
                         utils_0::annihilate_matrix(level_count));
    auto create_full = cudaq::kronecker(utils_0::create_matrix(level_count),
                                        utils_0::id_matrix(level_count));
    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = annihilate_full - create_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(0);

    auto product = self * other;
    ASSERT_TRUE(product.n_terms() == 2);

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = utils_0::annihilate_matrix(level_count) *
                       utils_0::create_matrix(level_count);
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto other = cudaq::matrix_operator::create(1);

    auto product = self * other;
    ASSERT_TRUE(product.n_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto annihilate_full =
        cudaq::kronecker(utils_0::id_matrix(level_count),
                         utils_0::annihilate_matrix(level_count));
    auto create_full = cudaq::kronecker(utils_0::create_matrix(level_count),
                                        utils_0::id_matrix(level_count));
    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = annihilate_full * create_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }
}

/// Testing arithmetic between elementary operators and operator
/// sums.
TEST(OperatorExpressions, checkElementaryOpsAgainstOpSum) {

  /// Keeping this fixed throughout.
  int level_count = 3;
  std::complex<double> value = 0.125 + 0.5j;

  /// `matrix_operator + operator_sum` and `operator_sum +
  /// matrix_operator`
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto operator_sum =
        cudaq::matrix_operator::create(0) + cudaq::matrix_operator::identity(1);

    auto got = self + operator_sum;
    auto reverse = operator_sum + self;

    ASSERT_TRUE(got.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto self_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                      utils_0::annihilate_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::create_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = self_full + term_0_full + term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full + self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  /// `matrix_operator - operator_sum` and `operator_sum -
  /// matrix_operator`
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto operator_sum =
        cudaq::matrix_operator::create(0) + cudaq::matrix_operator::identity(1);

    auto got = self - operator_sum;
    auto reverse = operator_sum - self;

    ASSERT_TRUE(got.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto self_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                      utils_0::annihilate_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::create_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = self_full - term_0_full - term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full - self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  /// `matrix_operator * operator_sum` and `operator_sum *
  /// matrix_operator`
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto operator_sum = cudaq::matrix_operator::squeeze(0) +
                        cudaq::matrix_operator::identity(1);

    auto got = self * operator_sum;
    auto reverse = operator_sum * self;

    ASSERT_TRUE(got.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);
    for (auto &term : got.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);
    for (auto &term : reverse.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);

    auto self_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                      utils_0::annihilate_matrix(level_count));
    auto term_0_full =
        cudaq::kronecker(utils_0::id_matrix(level_count),
                         utils_0::squeeze_matrix(level_count, value));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}},
                                    {{"squeezing", value}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{0, level_count}, {1, level_count}}, {{"squeezing", value}});
    auto want_matrix = self_full * sum_full;
    auto want_reverse_matrix = sum_full * self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
    utils_0::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  /// `operator_sum += matrix_operator`
  {
    auto operator_sum =
        cudaq::matrix_operator::create(0) + cudaq::matrix_operator::identity(1);
    operator_sum += cudaq::matrix_operator::displace(0);

    ASSERT_TRUE(operator_sum.n_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils_0::id_matrix(level_count),
                         utils_0::displace_matrix(level_count, value));
    auto term_0_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::create_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));

    auto got_matrix = operator_sum.to_matrix(
        {{0, level_count}, {1, level_count}}, {{"displacement", value}});
    auto want_matrix = term_0_full + term_1_full + self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  /// `operator_sum -= matrix_operator`
  {
    auto operator_sum =
        cudaq::matrix_operator::create(0) + cudaq::matrix_operator::identity(1);
    operator_sum -= cudaq::matrix_operator::annihilate(0);

    ASSERT_TRUE(operator_sum.n_terms() == 3);

    auto self_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                      utils_0::annihilate_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::create_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));

    auto got_matrix =
        operator_sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = term_0_full + term_1_full - self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }

  /// `operator_sum *= matrix_operator`
  {
    auto self = cudaq::matrix_operator::annihilate(0);
    auto operator_sum =
        cudaq::matrix_operator::create(0) + cudaq::matrix_operator::identity(1);

    operator_sum *= self;

    ASSERT_TRUE(operator_sum.n_terms() == 2);
    for (auto &term : operator_sum.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);

    auto self_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                      utils_0::annihilate_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::create_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils_0::id_matrix(level_count),
                                        utils_0::id_matrix(level_count));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix =
        operator_sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = sum_full * self_full;
    utils_0::checkEqual(want_matrix, got_matrix);
  }
}
