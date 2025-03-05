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

#include <numeric>

TEST(OperatorExpressions, checkProductOperatorBasics) {
  std::vector<int> levels = {2, 3, 4};

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  {// Same degrees of freedom.
   {auto spin0 = cudaq::spin_operator::x(5);
  auto spin1 = cudaq::spin_operator::z(5);
  auto spin_prod = spin0 * spin1;

  std::vector<int> want_degrees = {5};
  auto spin_matrix = utils::PauliX_matrix() * utils::PauliZ_matrix();

  ASSERT_TRUE(spin_prod.degrees() == want_degrees);
  utils::checkEqual(spin_matrix, spin_prod.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_operator::position(5);
    auto op1 = cudaq::matrix_operator::momentum(5);

    auto got = op0 * op1;
    utils::assert_product_equal(got, 1.,
                                {op0.get_terms()[0], op1.get_terms()[0]});
    ASSERT_TRUE(got.degrees() == want_degrees);

    auto got_matrix = got.to_matrix({{5, level_count}});
    auto matrix0 = utils::position_matrix(level_count);
    auto matrix1 = utils::momentum_matrix(level_count);
    auto want_matrix = matrix0 * matrix1;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

// Different degrees of freedom.
{
  auto spin0 = cudaq::spin_operator::x(0);
  auto spin1 = cudaq::spin_operator::z(1);
  auto spin_prod = spin0 * spin1;

  std::vector<int> want_degrees = {1, 0};
  auto spin_matrix =
      cudaq::kronecker(utils::PauliZ_matrix(), utils::PauliX_matrix());

  ASSERT_TRUE(spin_prod.degrees() == want_degrees);
  utils::checkEqual(spin_matrix, spin_prod.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_operator::position(0);
    auto op1 = cudaq::matrix_operator::momentum(1);

    cudaq::product_operator got = op0 * op1;
    cudaq::product_operator got_reverse = op1 * op0;

    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        got_reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto identity = utils::id_matrix(level_count);
    auto matrix0 = utils::position_matrix(level_count);
    auto matrix1 = utils::momentum_matrix(level_count);

    auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
    auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
    auto want_matrix = fullHilbert0 * fullHilbert1;
    auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

// Different degrees of freedom, non-consecutive.
// Should produce the same matrices as the above test.
{
  auto spin0 = cudaq::spin_operator::x(0);
  auto spin1 = cudaq::spin_operator::z(2);
  auto spin_prod = spin0 * spin1;

  std::vector<int> want_degrees = {2, 0};
  auto spin_matrix =
      cudaq::kronecker(utils::PauliZ_matrix(), utils::PauliX_matrix());

  ASSERT_TRUE(spin_prod.degrees() == want_degrees);
  utils::checkEqual(spin_matrix, spin_prod.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_operator::position(0);
    auto op1 = cudaq::matrix_operator::momentum(2);

    cudaq::product_operator got = op0 * op1;
    cudaq::product_operator got_reverse = op1 * op0;

    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);

    auto got_matrix = got.to_matrix({{0, level_count}, {2, level_count}});
    auto got_matrix_reverse =
        got_reverse.to_matrix({{0, level_count}, {2, level_count}});

    auto identity = utils::id_matrix(level_count);
    auto matrix0 = utils::position_matrix(level_count);
    auto matrix1 = utils::momentum_matrix(level_count);

    auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
    auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
    auto want_matrix = fullHilbert0 * fullHilbert1;
    auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

// Different degrees of freedom, non-consecutive but all dimensions
// provided.
{
  auto spin0 = cudaq::spin_operator::x(0);
  auto spin1 = cudaq::spin_operator::z(2);
  auto spin_prod = spin0 * spin1;

  std::vector<int> want_degrees = {2, 0};
  auto spin_matrix =
      cudaq::kronecker(utils::PauliZ_matrix(), utils::PauliX_matrix());
  std::unordered_map<int, int> dimensions = {{0, 2}, {1, 2}, {2, 2}};

  ASSERT_TRUE(spin_prod.degrees() == want_degrees);
  utils::checkEqual(spin_matrix, spin_prod.to_matrix(dimensions));

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_operator::position(0);
    auto op1 = cudaq::matrix_operator::momentum(2);

    cudaq::product_operator got = op0 * op1;
    cudaq::product_operator got_reverse = op1 * op0;

    std::vector<int> want_degrees = {2, 0};
    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);

    dimensions = {{0, level_count}, {1, level_count}, {2, level_count}};
    auto got_matrix = got.to_matrix(dimensions);
    auto got_matrix_reverse = got_reverse.to_matrix(dimensions);

    auto identity = utils::id_matrix(level_count);
    auto matrix0 = utils::position_matrix(level_count);
    auto matrix1 = utils::momentum_matrix(level_count);

    std::vector<cudaq::complex_matrix> matrices_0;
    std::vector<cudaq::complex_matrix> matrices_1;
    matrices_0 = {identity, matrix0};
    matrices_1 = {matrix1, identity};

    auto fullHilbert0 = cudaq::kronecker(matrices_0.begin(), matrices_0.end());
    auto fullHilbert1 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto want_matrix = fullHilbert0 * fullHilbert1;
    auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(got_matrix, want_matrix);
  }
}
}

// Scalar Ops against Elementary Ops
{
  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  // matrix operator against constant
  {
    auto op = cudaq::matrix_operator::position(0);
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto product = scalar_op * op;
    auto reverse = op * scalar_op;

    std::vector<int> want_degrees = {0};
    auto op_matrix = utils::position_matrix(2);

    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(value_0 * op_matrix, product.to_matrix({{0, 2}}));
    utils::checkEqual(value_0 * op_matrix, reverse.to_matrix({{0, 2}}));
  }

  // spin operator against constant
  {
    auto op = cudaq::spin_operator::x(0);
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto product = scalar_op * op;
    auto reverse = op * scalar_op;

    std::vector<int> want_degrees = {0};
    auto op_matrix = utils::PauliX_matrix();

    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(value_0 * op_matrix, product.to_matrix());
    utils::checkEqual(value_0 * op_matrix, reverse.to_matrix());
  }

  // matrix operator against constant from lambda
  {
    auto op = cudaq::matrix_operator::position(1);
    auto scalar_op = cudaq::scalar_operator(function);
    auto product = scalar_op * op;
    auto reverse = op * scalar_op;

    std::vector<int> want_degrees = {1};
    auto op_matrix = utils::position_matrix(2);

    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_op.evaluate({{"value", 0.3}}) * op_matrix,
                      product.to_matrix({{1, 2}}, {{"value", 0.3}}));
    utils::checkEqual(scalar_op.evaluate({{"value", 0.3}}) * op_matrix,
                      reverse.to_matrix({{1, 2}}, {{"value", 0.3}}));
  }

  // spin operator against constant from lambda
  {
    auto op = cudaq::spin_operator::x(1);
    auto scalar_op = cudaq::scalar_operator(function);
    auto product = scalar_op * op;
    auto reverse = op * scalar_op;

    std::vector<int> want_degrees = {1};
    auto op_matrix = utils::PauliX_matrix();

    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_op.evaluate({{"value", 0.3}}) * op_matrix,
                      product.to_matrix({}, {{"value", 0.3}}));
    utils::checkEqual(scalar_op.evaluate({{"value", 0.3}}) * op_matrix,
                      reverse.to_matrix({}, {{"value", 0.3}}));
  }
}
}

TEST(OperatorExpressions, checkProductOperatorAgainstScalars) {
  std::complex<double> value_0 = 0.1 + 0.1;
  int level_count = 3;

  /// `product_operator + double`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);

    auto sum = 2.0 + product_op;
    auto reverse = product_op + 2.0;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator + complex<double>`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);

    auto sum = value_0 + product_op;
    auto reverse = product_op + value_0;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `spin product + complex<double>`
  {
    auto product_op = cudaq::spin_operator::x(0) * cudaq::spin_operator::y(1);

    auto sum = value_0 + product_op;
    auto reverse = product_op + value_0;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto term_1 = cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto product = term_0 * term_1;
    auto scaled_identity = value_0 * utils::id_matrix(2 * 2);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator + scalar_operator`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto sum = scalar_op + product_op;
    auto reverse = product_op + scalar_op;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity + product;
    auto want_matrix_reverse = product + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - double`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);

    auto difference = 2.0 - product_op;
    auto reverse = product_op - 2.0;

    ASSERT_TRUE(difference.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(difference.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix =
        difference.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `spin product - double`
  {
    auto product_op = cudaq::spin_operator::i(0) * cudaq::spin_operator::z(1);

    auto sum = 2.0 - product_op;
    auto reverse = product_op - 2.0;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto term_1 = cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
    auto product = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(2 * 2);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - complex<double>`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);

    auto difference = value_0 - product_op;
    auto reverse = product_op - value_0;

    ASSERT_TRUE(difference.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(difference.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix =
        difference.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator - scalar_operator`
  {
    auto product_op = cudaq::matrix_operator::momentum(0) *
                      cudaq::matrix_operator::momentum(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto difference = scalar_op - product_op;
    auto reverse = product_op - scalar_op;

    ASSERT_TRUE(difference.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(difference.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix =
        difference.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::momentum_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity - product;
    auto want_matrix_reverse = product - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * double`
  {
    auto product_op =
        cudaq::matrix_operator::parity(0) * cudaq::matrix_operator::parity(1);
    ASSERT_TRUE(product_op.num_terms() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate() ==
                std::complex<double>(1.));

    auto product = 2.0 * product_op;
    auto reverse = product_op * 2.0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                std::complex<double>(2.));
    ASSERT_TRUE(reverse.get_coefficient().evaluate() ==
                std::complex<double>(2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::parity_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::parity_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * complex<double>`
  {
    auto product_op =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::number(1);
    ASSERT_TRUE(product_op.num_terms() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate() ==
                std::complex<double>(1.));

    auto product = value_0 * product_op;
    auto reverse = product_op * value_0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == value_0);
    ASSERT_TRUE(reverse.get_coefficient().evaluate() == value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::number_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator * scalar_operator`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto product = scalar_op * product_op;
    auto reverse = product_op * scalar_op;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == scalar_op.evaluate());
    ASSERT_TRUE(reverse.get_coefficient().evaluate() == scalar_op.evaluate());

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `spin product * scalar_operator`
  {
    auto product_op = cudaq::spin_operator::z(0) * cudaq::spin_operator::y(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto product = scalar_op * product_op;
    auto reverse = product_op * scalar_op;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == scalar_op.evaluate());
    ASSERT_TRUE(reverse.get_coefficient().evaluate() == scalar_op.evaluate());

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix = product.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliZ_matrix());
    auto term_1 = cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = value_0 * utils::id_matrix(2 * 2);

    auto want_matrix = scaled_identity * product_matrix;
    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator / double`
  {
    auto product_op =
        cudaq::matrix_operator::parity(0) * cudaq::matrix_operator::parity(1);
    ASSERT_TRUE(product_op.num_terms() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate() ==
                std::complex<double>(1.));

    auto reverse = product_op / 2.0;

    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(reverse.get_coefficient().evaluate() ==
                std::complex<double>(1. / 2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::parity_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::parity_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 0.5 * utils::id_matrix(level_count * level_count);

    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator / complex<double>`
  {
    auto product_op =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::number(1);
    ASSERT_TRUE(product_op.num_terms() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate() ==
                std::complex<double>(1.));

    auto reverse = product_op / value_0;

    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(reverse.get_coefficient().evaluate() == 1. / value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::number_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        (1. / value_0) * utils::id_matrix(level_count * level_count);

    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator / scalar_operator`
  {
    auto product_op = cudaq::matrix_operator::position(0) *
                      cudaq::matrix_operator::position(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto reverse = product_op / scalar_op;

    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(reverse.get_coefficient().evaluate() ==
                1. / scalar_op.evaluate());

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix_reverse =
        reverse.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::position_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        (1. / value_0) * utils::id_matrix(level_count * level_count);

    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `spin product / scalar_operator`
  {
    auto product_op = cudaq::spin_operator::z(0) * cudaq::spin_operator::y(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto reverse = product_op / scalar_op;

    ASSERT_TRUE(reverse.num_terms() == 2);
    ASSERT_TRUE(reverse.get_coefficient().evaluate() ==
                1. / scalar_op.evaluate());

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto got_matrix_reverse = reverse.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliZ_matrix());
    auto term_1 = cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = (1. / value_0) * utils::id_matrix(2 * 2);

    auto want_matrix_reverse = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  /// `product_operator *= double`
  {
    auto product = cudaq::matrix_operator::position(0) *
                   cudaq::matrix_operator::momentum(1);
    product *= 2.0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                std::complex<double>(2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `spin product *= double`
  {
    auto product = cudaq::spin_operator::y(0) * cudaq::spin_operator::i(1);
    product *= 2.0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                std::complex<double>(2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto term_1 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 2.0 * utils::id_matrix(2 * 2);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator *= complex<double>`
  {
    auto product =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::momentum(1);
    product *= value_0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator *= scalar_operator`
  {
    auto product =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::momentum(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    product *= scalar_op;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == scalar_op.evaluate());
    ASSERT_TRUE(scalar_op.evaluate() == value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        value_0 * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator /= double`
  {
    auto product = cudaq::matrix_operator::position(0) *
                   cudaq::matrix_operator::momentum(1);
    product /= 2.0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                std::complex<double>(1. / 2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::position_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 0.5 * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `spin product /= double`
  {
    auto product = cudaq::spin_operator::y(0) * cudaq::spin_operator::i(1);
    product /= 2.0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                std::complex<double>(1. / 2.));

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix();

    auto term_0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto term_1 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity = 0.5 * utils::id_matrix(2 * 2);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator /= complex<double>`
  {
    auto product =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::momentum(1);
    product /= value_0;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() == 1. / value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        (1. / value_0) * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  /// `product_operator /= scalar_operator`
  {
    auto product =
        cudaq::matrix_operator::number(0) * cudaq::matrix_operator::momentum(1);
    auto scalar_op = cudaq::scalar_operator(value_0);

    product /= scalar_op;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate() ==
                1. / scalar_op.evaluate());
    ASSERT_TRUE(scalar_op.evaluate() == value_0);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix({{0, level_count}, {1, level_count}});

    auto term_0 = cudaq::kronecker(utils::id_matrix(level_count),
                                   utils::number_matrix(level_count));
    auto term_1 = cudaq::kronecker(utils::momentum_matrix(level_count),
                                   utils::id_matrix(level_count));
    auto product_matrix = term_0 * term_1;
    auto scaled_identity =
        (1. / value_0) * utils::id_matrix(level_count * level_count);

    auto want_matrix = product_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstProduct) {

  int level_count = 3;
  std::unordered_map<int, int> dimensions = {
      {0, level_count}, {1, level_count}, {2, level_count + 1}};

  // `product_operator + product_operator`
  {
    auto term_0 = cudaq::matrix_operator::position(0) *
                  cudaq::matrix_operator::position(1);
    auto term_1 = cudaq::matrix_operator::momentum(1) *
                  cudaq::matrix_operator::position(2);

    auto sum = term_0 + term_1;

    ASSERT_TRUE(sum.num_terms() == 2);

    std::vector<int> want_degrees = {2, 1, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::position_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 1),
                    utils::position_matrix(level_count),
                    utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(level_count + 1),
                    utils::momentum_matrix(level_count),
                    utils::id_matrix(level_count)};
    matrices_1_1 = {utils::position_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix + term_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin product + spin product`
  {
    auto term_0 = cudaq::spin_operator::z(0) * cudaq::spin_operator::y(2);
    auto term_1 = cudaq::spin_operator::x(2) * cudaq::spin_operator::z(4);

    auto sum = term_0 + term_1;

    ASSERT_TRUE(sum.num_terms() == 2);

    std::vector<int> want_degrees = {4, 2, 0};
    ASSERT_TRUE(sum.degrees() == want_degrees);

    auto got_matrix = sum.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(2), utils::id_matrix(2),
                    utils::PauliZ_matrix()};
    matrices_0_1 = {utils::id_matrix(2), utils::PauliY_matrix(),
                    utils::id_matrix(2)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(2), utils::PauliX_matrix(),
                    utils::id_matrix(2)};
    matrices_1_1 = {utils::PauliZ_matrix(), utils::id_matrix(2),
                    utils::id_matrix(2)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix + term_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `product_operator - product_operator`
  {
    auto term_0 =
        cudaq::matrix_operator::position(0) * cudaq::matrix_operator::number(1);
    auto term_1 = cudaq::matrix_operator::momentum(1) *
                  cudaq::matrix_operator::momentum(2);

    auto difference = term_0 - term_1;

    ASSERT_TRUE(difference.num_terms() == 2);

    auto got_matrix = difference.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::position_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 1),
                    utils::number_matrix(level_count),
                    utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(level_count + 1),
                    utils::momentum_matrix(level_count),
                    utils::id_matrix(level_count)};
    matrices_1_1 = {utils::momentum_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix - term_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin product - spin product`
  {
    auto term_0 = cudaq::spin_operator::i(0);
    auto term_1 = cudaq::spin_operator::x(1) * cudaq::spin_operator::y(2);

    auto difference = term_0 - term_1;
    auto reverse = term_1 - term_0;

    ASSERT_TRUE(difference.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto got_matrix = difference.to_matrix();
    auto reverse_matrix = reverse.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0;
    matrices_0_0 = {utils::id_matrix(2), utils::id_matrix(2),
                    utils::id_matrix(2)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(2), utils::PauliX_matrix(),
                    utils::id_matrix(2)};
    matrices_1_1 = {utils::PauliY_matrix(), utils::id_matrix(2),
                    utils::id_matrix(2)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix - term_1_matrix;
    auto want_reverse_matrix = term_1_matrix - term_0_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, reverse_matrix);
  }

  // `product_operator * product_operator`
  {
    auto term_0 = cudaq::matrix_operator::position(0) *
                  cudaq::matrix_operator::position(1);
    auto term_1 =
        cudaq::matrix_operator::momentum(1) * cudaq::matrix_operator::parity(2);

    auto product = term_0 * term_1;

    ASSERT_TRUE(product.num_terms() == 4);

    auto got_matrix = product.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::position_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 1),
                    utils::position_matrix(level_count),
                    utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(level_count + 1),
                    utils::momentum_matrix(level_count),
                    utils::id_matrix(level_count)};
    matrices_1_1 = {utils::parity_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin product * spin product`
  {
    auto term_0 = cudaq::spin_operator::y(0) * cudaq::spin_operator::x(1);
    auto term_1 = cudaq::spin_operator::z(1) * cudaq::spin_operator::i(3);

    auto product = term_0 * term_1;
    auto reverse = term_1 * term_0;
    std::vector<int> expected_degrees = {3, 1, 0};

    ASSERT_TRUE(product.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);
    ASSERT_TRUE(product.degrees() == expected_degrees);

    auto got_matrix = product.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(2), utils::id_matrix(2),
                    utils::PauliY_matrix()};
    matrices_0_1 = {utils::id_matrix(2), utils::PauliX_matrix(),
                    utils::id_matrix(2)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(2), utils::PauliZ_matrix(),
                    utils::id_matrix(2)};
    matrices_1_1 = {utils::id_matrix(2), utils::id_matrix(2),
                    utils::id_matrix(2)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    auto want_reverse_matrix = term_1_matrix * term_0_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `product_operator *= product_operator`
  {
    auto term_0 =
        cudaq::matrix_operator::position(0) * cudaq::matrix_operator::number(1);
    auto term_1 = cudaq::matrix_operator::momentum(1) *
                  cudaq::matrix_operator::position(2);

    term_0 *= term_1;

    ASSERT_TRUE(term_0.num_terms() == 4);

    auto got_matrix = term_0.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::position_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 1),
                    utils::number_matrix(level_count),
                    utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(level_count + 1),
                    utils::momentum_matrix(level_count),
                    utils::id_matrix(level_count)};
    matrices_1_1 = {utils::position_matrix(level_count + 1),
                    utils::id_matrix(level_count),
                    utils::id_matrix(level_count)};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    auto term1_only_matrix =
        cudaq::kronecker(utils::position_matrix(level_count + 1),
                         utils::momentum_matrix(level_count));
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(term1_only_matrix, term_1.to_matrix(dimensions));
  }

  // `spin product *= spin product`
  {
    auto term_0 = cudaq::spin_operator::y(3) * cudaq::spin_operator::y(1);
    auto term_1 = cudaq::spin_operator::z(1) * cudaq::spin_operator::x(0);

    term_0 *= term_1;

    ASSERT_TRUE(term_0.num_terms() == 3);

    auto got_matrix = term_0.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    matrices_0_0 = {utils::PauliY_matrix(), utils::id_matrix(2),
                    utils::id_matrix(2)};
    matrices_0_1 = {utils::id_matrix(2), utils::PauliY_matrix(),
                    utils::id_matrix(2)};

    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    matrices_1_0 = {utils::id_matrix(2), utils::PauliZ_matrix(),
                    utils::id_matrix(2)};
    matrices_1_1 = {utils::id_matrix(2), utils::id_matrix(2),
                    utils::PauliX_matrix()};

    auto term_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto term_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) *
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = term_0_matrix * term_1_matrix;
    auto term1_only_matrix =
        cudaq::kronecker(utils::PauliZ_matrix(), utils::PauliX_matrix());
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(term1_only_matrix, term_1.to_matrix());
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstOperatorSum) {

  int level_count = 3;
  std::unordered_map<int, int> dimensions = {
      {0, level_count}, {1, level_count}, {2, level_count + 1}};

  // `product_operator + operator_sum`
  {
    auto product = cudaq::matrix_operator::position(0) *
                   cudaq::matrix_operator::position(1);
    auto original_sum = cudaq::matrix_operator::momentum(1) +
                        cudaq::matrix_operator::momentum(2);

    auto sum = product + original_sum;
    auto reverse = original_sum + product;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix(dimensions);
    auto got_matrix_reverse = reverse.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::position_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 1), utils::position_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 1), utils::momentum_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::momentum_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix + sum_matrix;
    auto want_matrix_reverse = sum_matrix + product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin product + spin sum`
  {
    auto product = cudaq::spin_operator::x(0) * cudaq::spin_operator::y(1);
    auto original_sum = cudaq::spin_operator::z(1) + cudaq::spin_operator::i(2);

    auto sum = product + original_sum;
    auto reverse = original_sum + product;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::PauliX_matrix()};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(2), utils::PauliY_matrix(), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(2), utils::PauliZ_matrix(), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::id_matrix(2)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix + sum_matrix;
    auto want_matrix_reverse = sum_matrix + product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator - operator_sum`
  {
    auto product = cudaq::matrix_operator::position(0) *
                   cudaq::matrix_operator::position(1);
    auto original_difference = cudaq::matrix_operator::momentum(1) -
                               cudaq::matrix_operator::momentum(2);

    auto difference = product - original_difference;
    auto reverse = original_difference - product;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix(dimensions);
    auto got_matrix_reverse = reverse.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::position_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 1), utils::position_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 1), utils::momentum_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::momentum_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto difference_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) -
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix - difference_matrix;
    auto want_matrix_reverse = difference_matrix - product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin product - spin sum`
  {
    auto product = cudaq::spin_operator::y(0) * cudaq::spin_operator::z(1);
    auto original_difference =
        cudaq::spin_operator::x(1) - cudaq::spin_operator::i(2);

    auto difference = product - original_difference;
    auto reverse = original_difference - product;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::PauliY_matrix()};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(2), utils::PauliZ_matrix(), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(2), utils::PauliX_matrix(), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::id_matrix(2)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto difference_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) -
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix - difference_matrix;
    auto want_matrix_reverse = difference_matrix - product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `product_operator * operator_sum`
  {
    auto original_product = cudaq::matrix_operator::position(0) *
                            cudaq::matrix_operator::position(1);
    auto sum = cudaq::matrix_operator::momentum(1) +
               cudaq::matrix_operator::momentum(2);

    auto product = original_product * sum;
    auto reverse = sum * original_product;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto got_matrix = product.to_matrix(dimensions);
    auto got_matrix_reverse = reverse.to_matrix(dimensions);

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::position_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 1), utils::position_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 1), utils::momentum_matrix(level_count),
        utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::momentum_matrix(level_count + 1), utils::id_matrix(level_count),
        utils::id_matrix(level_count)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix * sum_matrix;
    auto want_matrix_reverse = sum_matrix * product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin product * spin sum`
  {
    auto original_product =
        cudaq::spin_operator::z(0) * cudaq::spin_operator::y(1);
    auto sum = cudaq::spin_operator::i(1) + cudaq::spin_operator::x(2);

    auto product = original_product * sum;
    auto reverse = sum * original_product;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto got_matrix = product.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::PauliZ_matrix()};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(2), utils::PauliY_matrix(), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(2), utils::id_matrix(2), utils::id_matrix(2)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::PauliX_matrix(), utils::id_matrix(2), utils::id_matrix(2)};
    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = product_matrix * sum_matrix;
    auto want_matrix_reverse = sum_matrix * product_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

TEST(OperatorExpressions, checkCustomProductOps) {
  auto level_count = 2;
  std::unordered_map<int, int> dimensions = {{0, level_count + 1},
                                             {1, level_count + 2},
                                             {2, level_count},
                                             {3, level_count + 3}};

  {
    auto func0 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                  utils::position_matrix(dimensions[1]));
        };
    auto func1 =
        [](const std::vector<int> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                  utils::number_matrix(dimensions[1]));
        };
    cudaq::matrix_operator::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_operator::define("custom_op1", {-1, -1}, func1);
  }

  auto op0 = cudaq::matrix_operator::instantiate("custom_op0", {1, 0});
  auto op1 = cudaq::matrix_operator::instantiate("custom_op1", {2, 1});
  auto product = op0 * op1;
  auto reverse = op1 * op0;

  std::vector<cudaq::complex_matrix> matrices = {
      utils::momentum_matrix(level_count),
      utils::momentum_matrix(level_count + 2) *
          utils::number_matrix(level_count + 2),
      utils::position_matrix(level_count + 1)};
  auto expected = cudaq::kronecker(matrices.begin(), matrices.end());

  std::vector<cudaq::complex_matrix> matrices_reverse = {
      utils::momentum_matrix(level_count),
      utils::number_matrix(level_count + 2) *
          utils::momentum_matrix(level_count + 2),
      utils::position_matrix(level_count + 1)};
  auto expected_reverse =
      cudaq::kronecker(matrices_reverse.begin(), matrices_reverse.end());

  utils::checkEqual(product.to_matrix(dimensions), expected);
  utils::checkEqual(reverse.to_matrix(dimensions), expected_reverse);

  op0 = cudaq::matrix_operator::instantiate("custom_op0", {3, 2});
  op1 = cudaq::matrix_operator::instantiate("custom_op1", {2, 0});
  product = op0 * op1;
  reverse = op1 * op0;

  matrices = {utils::momentum_matrix(level_count + 3),
              utils::position_matrix(level_count) *
                  utils::momentum_matrix(level_count),
              utils::number_matrix(level_count + 1)};
  expected = cudaq::kronecker(matrices.begin(), matrices.end());

  matrices_reverse = {utils::momentum_matrix(level_count + 3),
                      utils::momentum_matrix(level_count) *
                          utils::position_matrix(level_count),
                      utils::number_matrix(level_count + 1)};
  expected_reverse =
      cudaq::kronecker(matrices_reverse.begin(), matrices_reverse.end());

  utils::checkEqual(product.to_matrix(dimensions), expected);
  utils::checkEqual(reverse.to_matrix(dimensions), expected_reverse);
}
