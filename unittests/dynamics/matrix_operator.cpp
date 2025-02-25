/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "utils.h"
#include <gtest/gtest.h>

TEST(OperatorExpressions, checkMatrixOpsUnary) {
  auto op = cudaq::matrix_operator::position(0);
  utils::checkEqual((+op).to_matrix({{0,2}}), utils::position_matrix(2));
  utils::checkEqual((-op).to_matrix({{0,2}}), -1.0 * utils::position_matrix(2));
  utils::checkEqual(op.to_matrix({{0,2}}), utils::position_matrix(2));
}

TEST(OperatorExpressions, checkMatrixOpsConstruction) {
  auto prod = cudaq::matrix_operator::identity();
  cudaq::matrix_2 expected(1, 1);

  expected[{0, 0}] = 1.;  
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= -1.j;
  expected[{0, 0}] = std::complex<double>(-1.j);
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= cudaq::matrix_operator::number(0);
  expected = cudaq::matrix_2(3, 3);
  expected[{1, 1}] = std::complex<double>(-1.j);
  expected[{2, 2}] = std::complex<double>(-2.j);
  utils::checkEqual(prod.to_matrix({{0, 3}}), expected);

  auto sum = cudaq::matrix_operator::empty();
  expected = cudaq::matrix_2(0, 0);
  utils::checkEqual(sum.to_matrix(), expected);

  sum *= cudaq::matrix_operator::number(1); // empty times something is still empty
  std::vector<int> expected_degrees = {};
  ASSERT_EQ(sum.degrees(), expected_degrees);
  utils::checkEqual(sum.to_matrix(), expected);

  sum += cudaq::matrix_operator::identity(1);
  expected = cudaq::matrix_2(3, 3);
  for (size_t i = 0; i < 3; ++i)
    expected[{i, i}] = 1.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum *= cudaq::matrix_operator::number(1);
  expected = cudaq::matrix_2(3, 3);
  expected[{1, 1}] = 1.;
  expected[{2, 2}] = 2.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum = cudaq::matrix_operator::empty();
  sum -= cudaq::matrix_operator::identity(0);
  expected = cudaq::matrix_2(3, 3);
  for (size_t i = 0; i < 3; ++i)
    expected[{i, i}] = -1.;
  utils::checkEqual(sum.to_matrix({{0, 3}}), expected);
}

TEST(OperatorExpressions, checkPreBuiltMatrixOps) {
  std::vector<std::size_t> levels = {2, 3, 4, 5};

  // Keeping this fixed throughout.
  int degree_index = 0;

  // Identity operator.
  {
    for (auto level_count : levels) {
      auto id = cudaq::matrix_operator::identity(degree_index);
      auto got_id = id.to_matrix({{degree_index, level_count}});
      auto want_id = utils::id_matrix(level_count);
      utils::checkEqual(want_id, got_id);
    }
  }

  // Number operator.
  {
    for (auto level_count : levels) {
      auto number = cudaq::matrix_operator::number(degree_index);
      auto got_number = number.to_matrix({{degree_index, level_count}});
      auto want_number = utils::number_matrix(level_count);
      utils::checkEqual(want_number, got_number);
    }
  }

  // Parity operator.
  {
    for (auto level_count : levels) {
      auto parity = cudaq::matrix_operator::parity(degree_index);
      auto got_parity = parity.to_matrix({{degree_index, level_count}});
      auto want_parity = utils::parity_matrix(level_count);
      utils::checkEqual(want_parity, got_parity);
    }
  }

  // Position operator.
  {
    for (auto level_count : levels) {
      auto position = cudaq::matrix_operator::position(degree_index);
      auto got_position = position.to_matrix({{degree_index, level_count}});
      auto want_position = utils::position_matrix(level_count);
      utils::checkEqual(want_position, got_position);
    }
  }

  // Momentum operator.
  {
    for (auto level_count : levels) {
      auto momentum = cudaq::matrix_operator::momentum(degree_index);
      auto got_momentum = momentum.to_matrix({{degree_index, level_count}});
      auto want_momentum = utils::momentum_matrix(level_count);
      utils::checkEqual(want_momentum, got_momentum);
    }
  }

  // Displacement operator.
  {
    for (auto level_count : levels) {
      auto displacement = 2.0 + 1.0j;
      auto displace = cudaq::matrix_operator::displace(degree_index);
      auto got_displace = displace.to_matrix({{degree_index, level_count}},
                                             {{"displacement", displacement}});
      auto want_displace = utils::displace_matrix(level_count, displacement);
      utils::checkEqual(want_displace, got_displace);
    }
  }

  // Squeeze operator.
  {
    for (auto level_count : levels) {
      auto squeezing = 2.0 + 1.0j;
      auto squeeze = cudaq::matrix_operator::squeeze(degree_index);
      auto got_squeeze = squeeze.to_matrix({{degree_index, level_count}},
                                           {{"squeezing", squeezing}});
      auto want_squeeze = utils::squeeze_matrix(level_count, squeezing);
      utils::checkEqual(want_squeeze, got_squeeze);
    }
  }
}

TEST(OperatorExpressions, checkCustomMatrixOps) {
  auto level_count = 2;
  std::unordered_map<int, int> dimensions = {
      {0, level_count + 1}, {1, level_count + 2}, {3, level_count}};

  {
    auto func0 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                  utils::position_matrix(dimensions[1]));
          ;
        };
    auto func1 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::position_matrix(dimensions[0]),
                                  utils::number_matrix(dimensions[1]));
          ;
        };
    cudaq::matrix_operator::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_operator::define("custom_op1", {-1, -1}, func1);
  }

  // op 0:
  // momentum level+1 on 0
  // position level+2 on 1
  // op 1:
  // number level on 3
  // create level+2 on 1
  auto op0 = cudaq::matrix_operator::instantiate("custom_op0", {0, 1});
  auto op1 = cudaq::matrix_operator::instantiate("custom_op1", {1, 3});

  auto matrix0 = cudaq::kronecker(utils::momentum_matrix(level_count + 1),
                                  utils::position_matrix(level_count + 2));
  auto matrix1 = cudaq::kronecker(utils::position_matrix(level_count + 2),
                                  utils::number_matrix(level_count));

  std::vector<cudaq::matrix_2> product_matrices = {
      utils::number_matrix(level_count),
      utils::position_matrix(level_count + 2) *
          utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::matrix_2> product_reverse_matrices = {
      utils::number_matrix(level_count),
      utils::position_matrix(level_count + 2) *
          utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::matrix_2> sum_matrices_term0 = {
      utils::id_matrix(level_count), utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::matrix_2> sum_matrices_term1 = {
      utils::number_matrix(level_count),
      utils::position_matrix(level_count + 2),
      utils::id_matrix(level_count + 1)};

  auto expected_product =
      cudaq::kronecker(product_matrices.begin(), product_matrices.end());
  auto expected_product_reverse = cudaq::kronecker(
      product_reverse_matrices.begin(), product_reverse_matrices.end());
  auto expected_sum_term0 =
      cudaq::kronecker(sum_matrices_term0.begin(), sum_matrices_term0.end());
  auto expected_sum_term1 =
      cudaq::kronecker(sum_matrices_term1.begin(), sum_matrices_term1.end());

  utils::checkEqual(op0.to_matrix(dimensions),
                    matrix0); // *not* in canonical order; order as defined in
                              // custom op definition
  utils::checkEqual(op1.to_matrix(dimensions),
                    matrix1); // *not* in canonical order; order as defined in
                              // custom op definition
  utils::checkEqual((op0 * op1).to_matrix(dimensions),
                    expected_product); // now reordered in canonical order
  utils::checkEqual(
      (op1 * op0).to_matrix(dimensions),
      expected_product_reverse); // now reordered in canonical order
  utils::checkEqual((op0 + op1).to_matrix(dimensions),
                    expected_sum_term0 +
                        expected_sum_term1); // now reordered in canonical order
  utils::checkEqual((op1 + op0).to_matrix(dimensions),
                    expected_sum_term0 +
                        expected_sum_term1); // now reordered in canonical order
}

TEST(OperatorExpressions, checkMatrixOpsWithComplex) {
  std::complex<double> value = 0.125 + 0.125j;

  // `matrix_operator` + `complex<double>` and `complex<double>` +
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::momentum(0);

    auto sum = value + elementary;
    auto reverse = elementary + value;

    auto got_matrix = sum.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils::id_matrix(3);
    auto want_matrix = scaled_identity + utils::momentum_matrix(3);
    auto want_matrix_reverse = utils::momentum_matrix(3) + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `matrix_operator` - `complex<double>` and `complex<double>` -
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::position(0);

    auto difference = value - elementary;
    auto reverse = elementary - value;

    auto got_matrix = difference.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils::id_matrix(3);
    auto want_matrix = scaled_identity - utils::position_matrix(3);
    auto want_matrix_reverse = utils::position_matrix(3) - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `matrix_operator` * `complex<double>` and `complex<double>` *
  // `matrix_operator`
  {
    auto elementary = cudaq::matrix_operator::number(0);

    auto product = value * elementary;
    auto reverse = elementary * value;

    auto got_matrix = product.to_matrix({{0, 3}});
    auto got_matrix_reverse = reverse.to_matrix({{0, 3}});

    auto scaled_identity = value * utils::id_matrix(3);
    auto want_matrix = scaled_identity * utils::number_matrix(3);
    auto want_matrix_reverse = utils::number_matrix(3) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

TEST(OperatorExpressions, checkMatrixOpsWithScalars) {

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  /// Keeping these fixed for these more simple tests.
  int level_count = 3;
  int degree_index = 0;
  double const_scale_factor = 2.0;

  // `matrix_operator + scalar_operator`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix = utils::momentum_matrix(level_count) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils::momentum_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator + scalar_operator`
  {
    auto self = cudaq::matrix_operator::parity(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}},
                                    {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::parity_matrix(level_count) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils::parity_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator - scalar_operator`
  {
    auto self = cudaq::matrix_operator::number(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix = utils::number_matrix(level_count) - scaled_identity;
    auto want_reverse_matrix =
        scaled_identity - utils::number_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator - scalar_operator`
  {
    auto self = cudaq::matrix_operator::position(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = sum.to_matrix({{degree_index, level_count}},
                                    {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::position_matrix(level_count) - scaled_identity;
    auto want_reverse_matrix =
        scaled_identity - utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator * scalar_operator`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto product = self * other;
    auto reverse = other * self;

    auto momentum = cudaq::matrix_operator::momentum(0).get_terms()[0];
    utils::assert_product_equal(product, const_scale_factor, {momentum});
    utils::assert_product_equal(reverse, const_scale_factor, {momentum});

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = product.to_matrix({{degree_index, level_count}});
    auto got_reverse_matrix = reverse.to_matrix({{degree_index, level_count}});
    auto want_matrix = utils::momentum_matrix(level_count) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils::momentum_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator * scalar_operator`
  {
    auto self = cudaq::matrix_operator::position(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(level_count);
    auto got_matrix = product.to_matrix({{degree_index, level_count}},
                                        {{"value", const_scale_factor}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{degree_index, level_count}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::position_matrix(level_count) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }
}

TEST(OperatorExpressions, checkMatrixOpsSimpleArithmetics) {

  /// Keeping this fixed throughout.
  int level_count = 3;
  std::unordered_map<int, int> dimensions = {{0, level_count},
                                             {1, level_count}};

  // Addition, same DOF.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) +
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto annihilate_full = cudaq::kronecker(
        utils::id_matrix(level_count), utils::momentum_matrix(level_count));
    auto create_full = cudaq::kronecker(utils::position_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = annihilate_full + create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) -
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto annihilate_full = cudaq::kronecker(
        utils::id_matrix(level_count), utils::momentum_matrix(level_count));
    auto create_full = cudaq::kronecker(utils::position_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = annihilate_full - create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(0);

    auto product = self * other;
    ASSERT_TRUE(product.num_terms() == 2);

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) *
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto other = cudaq::matrix_operator::position(1);

    auto product = self * other;
    ASSERT_TRUE(product.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto annihilate_full = cudaq::kronecker(
        utils::id_matrix(level_count), utils::momentum_matrix(level_count));
    auto create_full = cudaq::kronecker(utils::position_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = annihilate_full * create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkMatrixOpsAdvancedArithmetics) {

  /// Keeping this fixed throughout.
  int level_count = 3;
  std::complex<double> value = 0.125 + 0.5j;

  // `matrix_operator + operator_sum`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto operator_sum = cudaq::matrix_operator::position(0) +
                        cudaq::matrix_operator::identity(1);

    auto got = self + operator_sum;
    auto reverse = operator_sum + self;

    ASSERT_TRUE(got.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = self_full + term_0_full + term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator - operator_sum`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto operator_sum = cudaq::matrix_operator::position(0) +
                        cudaq::matrix_operator::identity(1);

    auto got = self - operator_sum;
    auto reverse = operator_sum - self;

    ASSERT_TRUE(got.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = self_full - term_0_full - term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `matrix_operator * operator_sum`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto operator_sum = cudaq::matrix_operator::squeeze(0) +
                        cudaq::matrix_operator::identity(1);

    auto got = self * operator_sum;
    auto reverse = operator_sum * self;

    ASSERT_TRUE(got.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    for (auto &term : got.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);
    for (auto &term : reverse.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(level_count),
                         utils::squeeze_matrix(level_count, value));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}},
                                    {{"squeezing", value}});
    auto got_reverse_matrix = reverse.to_matrix(
        {{0, level_count}, {1, level_count}}, {{"squeezing", value}});
    auto want_matrix = self_full * sum_full;
    auto want_reverse_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `operator_sum += matrix_operator`
  {
    auto operator_sum = cudaq::matrix_operator::position(0) +
                        cudaq::matrix_operator::identity(1);
    operator_sum += cudaq::matrix_operator::displace(0);

    ASSERT_TRUE(operator_sum.num_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils::id_matrix(level_count),
                         utils::displace_matrix(level_count, value));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix = operator_sum.to_matrix(
        {{0, level_count}, {1, level_count}}, {{"displacement", value}});
    auto want_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= matrix_operator`
  {
    auto operator_sum = cudaq::matrix_operator::position(0) +
                        cudaq::matrix_operator::identity(1);
    operator_sum -= cudaq::matrix_operator::momentum(0);

    ASSERT_TRUE(operator_sum.num_terms() == 3);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix =
        operator_sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= matrix_operator`
  {
    auto self = cudaq::matrix_operator::momentum(0);
    auto operator_sum = cudaq::matrix_operator::position(0) +
                        cudaq::matrix_operator::identity(1);

    operator_sum *= self;

    ASSERT_TRUE(operator_sum.num_terms() == 2);
    for (auto &term : operator_sum.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix =
        operator_sum.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkMatrixOpsDegreeVerification) {
  auto op1 = cudaq::matrix_operator::position(2);
  auto op2 = cudaq::matrix_operator::momentum(0);
  std::unordered_map<int, int> dimensions = {{0, 2}, {1, 2}, {2, 3}, {3, 3}};

  {
    auto func0 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                  utils::position_matrix(dimensions[1]));
          ;
        };
    auto func1 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::position_matrix(dimensions[0]),
                                  utils::number_matrix(dimensions[1]));
          ;
        };
    cudaq::matrix_operator::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_operator::define("custom_op1", {-1, -1}, func1);
  }

  auto custom_op0 = cudaq::matrix_operator::instantiate("custom_op0", {3, 1});
  auto custom_op1 = cudaq::matrix_operator::instantiate("custom_op1", {1, 0});

  ASSERT_THROW(op1.to_matrix(), std::runtime_error);
  ASSERT_THROW(op1.to_matrix({{1, 2}}), std::runtime_error);
  ASSERT_THROW((op1 * op2).to_matrix({{2, 3}}), std::runtime_error);
  ASSERT_THROW((op1 + op2).to_matrix({{0, 3}}), std::runtime_error);
  ASSERT_NO_THROW((op1 * op2).to_matrix(dimensions));
  ASSERT_NO_THROW((op1 + op2).to_matrix(dimensions));

  ASSERT_THROW(custom_op0.to_matrix(), std::runtime_error);
  ASSERT_THROW(custom_op1.to_matrix({{1, 2}}), std::runtime_error);
  ASSERT_THROW((custom_op1 * custom_op0).to_matrix({{0, 2}, {1, 2}}),
               std::runtime_error);
  ASSERT_THROW((custom_op1 + custom_op0).to_matrix({{0, 2}, {1, 2}, {2, 2}}),
               std::runtime_error);
  ASSERT_NO_THROW((custom_op0 * custom_op1).to_matrix(dimensions));
  ASSERT_NO_THROW((custom_op0 + custom_op1).to_matrix(dimensions));
}

TEST(OperatorExpressions, checkMatrixOpsParameterVerification) {

  std::unordered_map<std::string, std::complex<double>> parameters = {
      {"squeezing", 0.5}, {"displacement", 0.25}};
  std::unordered_map<int, int> dimensions = {{0, 2}, {1, 2}};

  auto squeeze = cudaq::matrix_operator::squeeze(1);
  auto displace = cudaq::matrix_operator::displace(0);

  ASSERT_THROW(squeeze.to_matrix(dimensions), std::runtime_error);
  ASSERT_THROW(squeeze.to_matrix(dimensions, {{"displacement", 0.25}}),
               std::runtime_error);
  ASSERT_THROW(
      (squeeze * displace).to_matrix(dimensions, {{"displacement", 0.25}}),
      std::runtime_error);
  ASSERT_THROW((squeeze + displace).to_matrix(dimensions, {{"squeezing", 0.5}}),
               std::runtime_error);
  ASSERT_NO_THROW((squeeze * displace).to_matrix(dimensions, parameters));
  ASSERT_NO_THROW((squeeze + displace).to_matrix(dimensions, parameters));
}
