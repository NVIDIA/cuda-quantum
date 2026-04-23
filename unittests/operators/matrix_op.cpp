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

TEST(OperatorExpressions, checkMatrixOpsUnary) {
  auto op = cudaq::matrix_op::position(0);
  utils::checkEqual((+op).to_matrix({{0, 2}}), utils::position_matrix(2));
  utils::checkEqual((-op).to_matrix({{0, 2}}),
                    -1.0 * utils::position_matrix(2));
  utils::checkEqual(op.to_matrix({{0, 2}}), utils::position_matrix(2));
}

TEST(OperatorExpressions, checkMatrixOpsConstruction) {
  auto prod = cudaq::matrix_op::identity();
  cudaq::complex_matrix expected(1, 1);

  expected[{0, 0}] = 1.;
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= std::complex<double>(0., -1.);
  expected[{0, 0}] = std::complex<double>(0., -1.);
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= cudaq::matrix_op::number(0);
  expected = cudaq::complex_matrix(3, 3);
  expected[{1, 1}] = std::complex<double>(0., -1.);
  expected[{2, 2}] = std::complex<double>(0., -2.);
  utils::checkEqual(prod.to_matrix({{0, 3}}), expected);

  auto sum = cudaq::matrix_op::empty();
  expected = cudaq::complex_matrix(0, 0);
  utils::checkEqual(sum.to_matrix(), expected);

  sum *= cudaq::matrix_op::number(1); // empty times something is still empty
  std::vector<std::size_t> expected_degrees = {};
  ASSERT_EQ(sum.degrees(), expected_degrees);
  utils::checkEqual(sum.to_matrix(), expected);

  sum += cudaq::matrix_op::identity(1);
  expected = cudaq::complex_matrix(3, 3);
  for (size_t i = 0; i < 3; ++i)
    expected[{i, i}] = 1.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum *= cudaq::matrix_op::number(1);
  expected = cudaq::complex_matrix(3, 3);
  expected[{1, 1}] = 1.;
  expected[{2, 2}] = 2.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum = cudaq::matrix_op::empty();
  sum -= cudaq::matrix_op::identity(0);
  expected = cudaq::complex_matrix(3, 3);
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
      auto id = cudaq::matrix_op::identity(degree_index);
      auto got_id = id.to_matrix({{degree_index, level_count}});
      auto want_id = utils::id_matrix(level_count);
      utils::checkEqual(want_id, got_id);
    }
  }

  // Number operator.
  {
    for (auto level_count : levels) {
      auto number = cudaq::matrix_op::number(degree_index);
      auto got_number = number.to_matrix({{degree_index, level_count}});
      auto want_number = utils::number_matrix(level_count);
      utils::checkEqual(want_number, got_number);
    }
  }

  // Parity operator.
  {
    for (auto level_count : levels) {
      auto parity = cudaq::matrix_op::parity(degree_index);
      auto got_parity = parity.to_matrix({{degree_index, level_count}});
      auto want_parity = utils::parity_matrix(level_count);
      utils::checkEqual(want_parity, got_parity);
    }
  }

  // Position operator.
  {
    for (auto level_count : levels) {
      auto position = cudaq::matrix_op::position(degree_index);
      auto got_position = position.to_matrix({{degree_index, level_count}});
      auto want_position = utils::position_matrix(level_count);
      utils::checkEqual(want_position, got_position);
    }
  }

  // Momentum operator.
  {
    for (auto level_count : levels) {
      auto momentum = cudaq::matrix_op::momentum(degree_index);
      auto got_momentum = momentum.to_matrix({{degree_index, level_count}});
      auto want_momentum = utils::momentum_matrix(level_count);
      utils::checkEqual(want_momentum, got_momentum);
    }
  }

  // Displacement operator.
  {
    for (auto level_count : levels) {
      auto displacement = std::complex<double>(2.0, 1.0);
      auto displace = cudaq::matrix_op::displace(degree_index);
      auto got_displace = displace.to_matrix({{degree_index, level_count}},
                                             {{"displacement", displacement}});
      auto want_displace = utils::displace_matrix(level_count, displacement);
      utils::checkEqual(want_displace, got_displace);
    }
  }

  // Squeeze operator.
  {
    for (auto level_count : levels) {
      auto squeezing = std::complex<double>(2.0, 1.0);
      auto squeeze = cudaq::matrix_op::squeeze(degree_index);
      auto got_squeeze = squeeze.to_matrix({{degree_index, level_count}},
                                           {{"squeezing", squeezing}});
      auto want_squeeze = utils::squeeze_matrix(level_count, squeezing);
      utils::checkEqual(want_squeeze, got_squeeze);
    }
  }
}

TEST(OperatorExpressions, checkCustomMatrixOps) {
  auto level_count = 2;
  cudaq::dimension_map dimensions = {
      {0, level_count + 1}, {1, level_count + 2}, {3, level_count}};

  {
    auto func0 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::position_matrix(dimensions[1]),
                                  utils::momentum_matrix(dimensions[0]));
          ;
        };
    auto func1 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::number_matrix(dimensions[1]),
                                  utils::position_matrix(dimensions[0]));
          ;
        };
    cudaq::matrix_handler::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_handler::define("custom_op1", {-1, -1}, func1);
  }

  // check that we force user facing conventions when defining/instantiating
  // a custom operator
  ASSERT_ANY_THROW(cudaq::matrix_handler::instantiate("custom_op0", {1, 0}));
  ASSERT_ANY_THROW(cudaq::matrix_handler::instantiate("custom_op0", {3, 1}));

  // op 0:
  // momentum level+1 on 0
  // position level+2 on 1
  // op 1:
  // number level on 3
  // create level+2 on 1
  auto op0 = cudaq::matrix_handler::instantiate("custom_op0", {0, 1});
  auto op1 = cudaq::matrix_handler::instantiate("custom_op1", {1, 3});

  auto matrix0 = cudaq::kronecker(utils::position_matrix(level_count + 2),
                                  utils::momentum_matrix(level_count + 1));
  auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count),
                                  utils::position_matrix(level_count + 2));

  std::vector<cudaq::complex_matrix> product_matrices = {
      utils::number_matrix(level_count),
      utils::position_matrix(level_count + 2) *
          utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::complex_matrix> product_reverse_matrices = {
      utils::number_matrix(level_count),
      utils::position_matrix(level_count + 2) *
          utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::complex_matrix> sum_matrices_term0 = {
      utils::id_matrix(level_count), utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
  std::vector<cudaq::complex_matrix> sum_matrices_term1 = {
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

  utils::checkEqual(
      op0.to_matrix(dimensions),
      matrix0); // reordered to match CUDA-Q conventions by default
  utils::checkEqual(
      op1.to_matrix(dimensions),
      matrix1); // reordered to match CUDA-Q conventions by default
  utils::checkEqual((op0 * op1).to_matrix(dimensions), expected_product);
  utils::checkEqual((op1 * op0).to_matrix(dimensions),
                    expected_product_reverse);
  utils::checkEqual((op0 + op1).to_matrix(dimensions),
                    expected_sum_term0 + expected_sum_term1);
  utils::checkEqual((op1 + op0).to_matrix(dimensions),
                    expected_sum_term0 + expected_sum_term1);
}

TEST(OperatorExpressions, checkMatrixOpsWithComplex) {
  std::complex<double> value = std::complex<double>(0.125, 0.125);

  // `matrix_handler` + `complex<double>` and `complex<double>` +
  // `matrix_handler`
  {
    auto elementary = cudaq::matrix_op::momentum(0);

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

  // `matrix_handler` - `complex<double>` and `complex<double>` -
  // `matrix_handler`
  {
    auto elementary = cudaq::matrix_op::position(0);

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

  // `matrix_handler` * `complex<double>` and `complex<double>` *
  // `matrix_handler`
  {
    auto elementary = cudaq::matrix_op::number(0);

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

  // `matrix_handler + scalar_operator`
  {
    auto self = cudaq::matrix_op::momentum(0);
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

  // `matrix_handler + scalar_operator`
  {
    auto self = cudaq::matrix_op::parity(0);
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

  // `matrix_handler - scalar_operator`
  {
    auto self = cudaq::matrix_op::number(0);
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

  // `matrix_handler - scalar_operator`
  {
    auto self = cudaq::matrix_op::position(0);
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

  // `matrix_handler * scalar_operator`
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto product = self * other;
    auto reverse = other * self;

    auto momentum = *cudaq::matrix_op::momentum(0).begin();
    utils::assert_product_equal(product, const_scale_factor, {momentum});
    utils::assert_product_equal(reverse, const_scale_factor, {momentum});

    std::vector<std::size_t> want_degrees = {0};
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

  // `matrix_handler * scalar_operator`
  {
    auto self = cudaq::matrix_op::position(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<std::size_t> want_degrees = {0};
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
  cudaq::dimension_map dimensions = {{0, level_count}, {1, level_count}};

  // Addition, same DOF.
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) +
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(1);

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
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) -
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(1);

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
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(0);

    auto product = self * other;
    ASSERT_TRUE(product.num_ops() == 2);

    std::vector<std::size_t> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = utils::momentum_matrix(level_count) *
                       utils::position_matrix(level_count);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto other = cudaq::matrix_op::position(1);

    auto product = self * other;
    ASSERT_TRUE(product.num_ops() == 2);

    std::vector<std::size_t> want_degrees = {0, 1};
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
  std::complex<double> value = std::complex<double>(0.125, 0.5);

  // `matrix_handler + sum_op`
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto sum_op = cudaq::matrix_op::position(0) + cudaq::matrix_op::identity(1);

    auto got = self + sum_op;
    auto reverse = sum_op + self;

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

  // `matrix_handler - sum_op`
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto sum_op = cudaq::matrix_op::position(0) + cudaq::matrix_op::identity(1);

    auto got = self - sum_op;
    auto reverse = sum_op - self;

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

  // `matrix_handler * sum_op`
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto sum_op = cudaq::matrix_op::squeeze(0) + cudaq::matrix_op::identity(1);

    auto got = self * sum_op;
    auto reverse = sum_op * self;

    ASSERT_TRUE(got.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    for (const auto &term : got)
      ASSERT_TRUE(term.num_ops() == 2);
    for (const auto &term : reverse)
      ASSERT_TRUE(term.num_ops() == 2);

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

  // `sum_op += matrix_handler`
  {
    auto sum_op = cudaq::matrix_op::position(0) + cudaq::matrix_op::identity(1);
    sum_op += cudaq::matrix_op::displace(0);

    ASSERT_TRUE(sum_op.num_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils::id_matrix(level_count),
                         utils::displace_matrix(level_count, value));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix = sum_op.to_matrix({{0, level_count}, {1, level_count}},
                                       {{"displacement", value}});
    auto want_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= matrix_handler`
  {
    auto sum_op = cudaq::matrix_op::position(0) + cudaq::matrix_op::identity(1);
    sum_op -= cudaq::matrix_op::momentum(0);

    ASSERT_TRUE(sum_op.num_terms() == 3);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));

    auto got_matrix = sum_op.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= matrix_handler`
  {
    auto self = cudaq::matrix_op::momentum(0);
    auto sum_op = cudaq::matrix_op::position(0) + cudaq::matrix_op::identity(1);

    sum_op *= self;

    ASSERT_TRUE(sum_op.num_terms() == 2);
    for (const auto &term : sum_op)
      ASSERT_TRUE(term.num_ops() == 2);

    auto self_full = cudaq::kronecker(utils::id_matrix(level_count),
                                      utils::momentum_matrix(level_count));
    auto term_0_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::position_matrix(level_count));
    auto term_1_full = cudaq::kronecker(utils::id_matrix(level_count),
                                        utils::id_matrix(level_count));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = sum_op.to_matrix({{0, level_count}, {1, level_count}});
    auto want_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkMatrixOpsDegreeVerification) {
  auto op1 = cudaq::matrix_op::position(2);
  auto op2 = cudaq::matrix_op::momentum(0);
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}, {2, 3}, {3, 3}};

  {
    auto func0 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                  utils::position_matrix(dimensions[1]));
          ;
        };
    auto func1 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::position_matrix(dimensions[0]),
                                  utils::number_matrix(dimensions[1]));
          ;
        };
    cudaq::matrix_handler::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_handler::define("custom_op1", {-1, -1}, func1);

    cudaq::matrix_handler::define("custom_op2", {3, 3}, func0);
    cudaq::matrix_handler::define("custom_op3", {-1, 2}, func1);
  }

  auto custom_op0 = cudaq::matrix_handler::instantiate("custom_op0", {1, 3});
  auto custom_op1 = cudaq::matrix_handler::instantiate("custom_op1", {0, 1});

  ASSERT_ANY_THROW(op1.to_matrix());
  ASSERT_ANY_THROW(op1.to_matrix({{1, 2}}));
  ASSERT_ANY_THROW((op1 * op2).to_matrix({{2, 3}}));
  ASSERT_ANY_THROW((op1 + op2).to_matrix({{0, 3}}));
  ASSERT_NO_THROW((op1 * op2).to_matrix(dimensions));
  ASSERT_NO_THROW((op1 + op2).to_matrix(dimensions));

  ASSERT_ANY_THROW(custom_op0.to_matrix());
  ASSERT_ANY_THROW(custom_op1.to_matrix({{1, 2}}));
  ASSERT_ANY_THROW((custom_op1 * custom_op0).to_matrix({{0, 2}, {1, 2}}));
  ASSERT_ANY_THROW(
      (custom_op1 + custom_op0).to_matrix({{0, 2}, {1, 2}, {2, 2}}));
  ASSERT_NO_THROW((custom_op0 * custom_op1).to_matrix(dimensions));
  ASSERT_NO_THROW((custom_op0 + custom_op1).to_matrix(dimensions));

  auto custom_op2 = cudaq::matrix_handler::instantiate("custom_op2", {1, 3});
  auto custom_op3 = cudaq::matrix_handler::instantiate("custom_op3", {0, 1});

  dimensions = {{0, 2}};
  ASSERT_NO_THROW(custom_op2.to_matrix());
  ASSERT_ANY_THROW(custom_op3.to_matrix());
  ASSERT_NO_THROW(custom_op3.to_matrix(dimensions));
  dimensions = {};
  ASSERT_NO_THROW(custom_op2.to_matrix(
      dimensions)); // degree 1 should be set to required dim 3
  ASSERT_ANY_THROW(custom_op3.to_matrix(dimensions)); // degree 1 needs to be 2
}

TEST(OperatorExpressions, checkMatrixOpsParameterVerification) {

  std::unordered_map<std::string, std::complex<double>> parameters = {
      {"squeezing", 0.5}, {"displacement", 0.25}};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};

  auto squeeze = cudaq::matrix_op::squeeze(1);
  auto displace = cudaq::matrix_op::displace(0);

  ASSERT_ANY_THROW(squeeze.to_matrix(dimensions));
  ASSERT_ANY_THROW(squeeze.to_matrix(dimensions, {{"displacement", 0.25}}));
  ASSERT_ANY_THROW(
      (squeeze * displace).to_matrix(dimensions, {{"displacement", 0.25}}));
  ASSERT_ANY_THROW(
      (squeeze + displace).to_matrix(dimensions, {{"squeezing", 0.5}}));
  ASSERT_NO_THROW((squeeze * displace).to_matrix(dimensions, parameters));
  ASSERT_NO_THROW((squeeze + displace).to_matrix(dimensions, parameters));
}
