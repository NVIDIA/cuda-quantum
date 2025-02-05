/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "utils.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>

TEST(OperatorExpressions, checkOperatorSumAgainstScalarOperator) {
  int level_count = 3;
  std::complex<double> value = 0.2 + 0.2j;

  // `operator_sum * scalar_operator` and `scalar_operator * operator_sum`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum + scalar_operator` and `scalar_operator + operator_sum`
  {
    level_count = 2;
    auto original = cudaq::matrix_operator::create(1) +
                    cudaq::matrix_operator::create(2);

    auto sum = original + cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) + original;

    ASSERT_TRUE(sum.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);


    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2,level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - scalar_operator` and `scalar_operator - operator_sum`
  {
    auto original = cudaq::matrix_operator::create(1) +
                    cudaq::matrix_operator::create(2);

    auto difference = original - cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) - original;

    ASSERT_TRUE(difference.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto got_matrix = difference.to_matrix({{1, level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= scalar_operator`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::momentum(2);

    sum *= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.n_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}, {2, level_count+1}});

    std::vector<cudaq::matrix_2> matrices_1 = {
        utils::id_matrix(level_count + 1),
        utils::create_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_2 = {
        utils::momentum_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += scalar_operator`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::position(2);

    sum += cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}, {2, level_count+1}});

    std::vector<cudaq::matrix_2> matrices_1 = {
        utils::id_matrix(level_count + 1),
        utils::parity_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_2 = {
        utils::position_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = matrix0 + matrix1 + scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= scalar_operator`
  {
    auto sum = cudaq::matrix_operator::number(1) +
               cudaq::matrix_operator::annihilate(2);

    sum -= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}, {2, level_count+1}});

    std::vector<cudaq::matrix_2> matrices_1 = {
        utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_2 = {
        utils::annihilate_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkOperatorSumAgainstScalars) {
  int level_count = 3;
  std::complex<double> value = 0.1 + 0.1j;
  double double_value = 0.1;

  // `operator_sum * double` and `double * operator_sum`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    auto product = sum * double_value;
    auto reverse = double_value * sum;

    ASSERT_TRUE(product.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    auto got_matrix = product.to_matrix({{1, level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum + double` and `double + operator_sum`
  {
    auto original = cudaq::matrix_operator::momentum(1) +
                    cudaq::matrix_operator::position(2);

    auto sum = original + double_value;
    auto reverse = double_value + original;

    ASSERT_TRUE(sum.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::position_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - double` and `double - operator_sum`
  {
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::number(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto got_matrix = difference.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= double`
  {
    auto sum = cudaq::matrix_operator::squeeze(1) +
               cudaq::matrix_operator::squeeze(2);

    sum *= double_value;

    ASSERT_TRUE(sum.n_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}}, {{"squeezing", value}});

    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(level_count + 1),
                         utils::squeeze_matrix(level_count, value));
    auto matrix1 =
        cudaq::kronecker(utils::squeeze_matrix(level_count + 1, value),
                         utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += double`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    sum += double_value;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= double`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    sum -= double_value;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum * std::complex<double>` and `std::complex<double> *
  // operator_sum`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    auto product = sum * value;
    auto reverse = value * sum;

    ASSERT_TRUE(product.n_terms() == 2);
    ASSERT_TRUE(reverse.n_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = product.to_matrix({{1,level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum + std::complex<double>` and `std::complex<double> +
  // operator_sum`
  {
    auto original = cudaq::matrix_operator::create(1) +
                    cudaq::matrix_operator::create(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - std::complex<double>` and `std::complex<double> -
  // operator_sum`
  {
    auto original = cudaq::matrix_operator::create(1) +
                    cudaq::matrix_operator::create(2);

    auto difference = original - value;
    auto reverse = value - original;

    ASSERT_TRUE(difference.n_terms() == 3);
    ASSERT_TRUE(reverse.n_terms() == 3);

    auto got_matrix = difference.to_matrix({{1,level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::create_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::create_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::displace(1) +
               cudaq::matrix_operator::parity(2);

    sum *= value;

    ASSERT_TRUE(sum.n_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}}, {{"displacement", value}});

    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(level_count + 1),
                         utils::displace_matrix(level_count, value));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::momentum(1) +
               cudaq::matrix_operator::squeeze(2);

    sum += value;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}}, {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 =
        cudaq::kronecker(utils::squeeze_matrix(level_count + 1, value),
                         utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::position(1) +
               cudaq::matrix_operator::number(2);

    sum -= value;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::position_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkOperatorSumAgainstOperatorSum) {
  int level_count = 2;

  // `operator_sum + operator_sum`
  {
    auto sum_0 = cudaq::matrix_operator::create(1) +
                 cudaq::matrix_operator::create(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::annihilate(1) +
                 cudaq::matrix_operator::create(3);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.n_terms() == 5);

    auto got_matrix = sum.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::create_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::create_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::annihilate_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::create_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix + sum_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum - operator_sum`
  {
    auto sum_0 = cudaq::matrix_operator::create(1) +
                 cudaq::matrix_operator::position(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::annihilate(1) +
                 cudaq::matrix_operator::momentum(3);

    auto difference = sum_0 - sum_1;

    ASSERT_TRUE(difference.n_terms() == 5);

    auto got_matrix = difference.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::create_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::position_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::annihilate_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::momentum_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix - sum_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum * operator_sum`
  {
    auto sum_0 = cudaq::matrix_operator::create(1) +
                 cudaq::matrix_operator::create(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::annihilate(1) +
                 cudaq::matrix_operator::create(3);

    auto sum_product = sum_0 * sum_1;
    auto sum_product_reverse = sum_1 * sum_0;

    ASSERT_TRUE(sum_product.n_terms() == 6);
    ASSERT_TRUE(sum_product_reverse.n_terms() == 6);
    for (auto term : sum_product.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);
    for (auto term : sum_product_reverse.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);

    auto got_matrix = sum_product.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}}); 
    auto got_matrix_reverse = sum_product_reverse.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::create_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::create_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::annihilate_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::create_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix * sum_1_matrix;
    auto want_matrix_reverse = sum_1_matrix * sum_0_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum *= operator_sum`
  {
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::annihilate(1) +
                 cudaq::matrix_operator::create(3);

    sum *= sum_1;

    ASSERT_TRUE(sum.n_terms() == 6);
    for (auto term : sum.get_terms())
      ASSERT_TRUE(term.n_terms() == 2);

    auto got_matrix = sum.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::create_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::create_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::annihilate_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::create_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};

    auto sum_0_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) +
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_1_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end()) +
        cudaq::kronecker(matrices_1_2.begin(), matrices_1_2.end());

    auto want_matrix = sum_0_matrix * sum_1_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

/// NOTE: Much of the simpler arithmetic between the two is tested in the
/// product operator test file. This mainly just tests the assignment operators
/// between the two types.
TEST(OperatorExpressions, checkOperatorSumAgainstProduct) {
  int level_count = 2;

  // `operator_sum += product_operator`
  {
    auto product = cudaq::matrix_operator::annihilate(0) *
                   cudaq::matrix_operator::annihilate(1);
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    sum += product;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::annihilate_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::create_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::create_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), 
        utils::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix + product_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= product_operator`
  {
    auto product = cudaq::matrix_operator::annihilate(0) *
                   cudaq::matrix_operator::annihilate(1);
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    sum -= product;

    ASSERT_TRUE(sum.n_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::annihilate_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::create_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::create_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), 
        utils::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix - product_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= product_operator`
  {
    auto product = cudaq::matrix_operator::annihilate(0) *
                   cudaq::matrix_operator::annihilate(1);
    auto sum = cudaq::matrix_operator::create(1) +
               cudaq::matrix_operator::create(2);

    sum *= product;

    ASSERT_TRUE(sum.n_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.n_terms() == 3);
    }

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::annihilate_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::annihilate_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::create_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::create_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), 
        utils::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix * product_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkCustomOperatorSum) {
    auto level_count = 2;
    std::map<int, int> dimensions = {{0, level_count + 1}, {1, level_count + 2}, {2, level_count}, {3, level_count + 3}};

    {
      auto func0 = [](std::vector<int> dimensions,
                      std::map<std::string, std::complex<double>> _none) {
        return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                      utils::position_matrix(dimensions[1]));;
      };
      auto func1 = [](std::vector<int> dimensions,
                      std::map<std::string, std::complex<double>> _none) {
        return cudaq::kronecker(utils::create_matrix(dimensions[0]),
                                      utils::number_matrix(dimensions[1]));;
      };
      cudaq::matrix_operator::define("custom_op0", {-1, -1}, func0);
      cudaq::matrix_operator::define("custom_op1", {-1, -1}, func1);
    }

    auto op0 = cudaq::product_operator<cudaq::matrix_operator>(1., cudaq::matrix_operator("custom_op0", {0, 1}));
    auto op1 = cudaq::product_operator<cudaq::matrix_operator>(1., cudaq::matrix_operator("custom_op1", {1, 2}));
    auto sum = op0 + op1;
    auto sum_reverse = op1 + op0;
    auto difference = op0 - op1;
    auto difference_reverse = op1 - op0;

    std::vector<cudaq::matrix_2> matrices_0 = {
      utils::id_matrix(level_count),
      utils::position_matrix(level_count + 2),
      utils::momentum_matrix(level_count + 1)};
    std::vector<cudaq::matrix_2> matrices_1 = {
      utils::number_matrix(level_count),
      utils::create_matrix(level_count + 2),
      utils::id_matrix(level_count + 1)};
    auto sum_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) + 
                        cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto diff_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) - 
                         cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto diff_reverse_expected = cudaq::kronecker(matrices_1.begin(), matrices_1.end()) -
                                 cudaq::kronecker(matrices_0.begin(), matrices_0.end());

    utils::checkEqual(sum.to_matrix(dimensions), sum_expected);
    utils::checkEqual(sum_reverse.to_matrix(dimensions), sum_expected);
    utils::checkEqual(difference.to_matrix(dimensions), diff_expected);
    utils::checkEqual(difference_reverse.to_matrix(dimensions), diff_reverse_expected);

    op0 = cudaq::product_operator<cudaq::matrix_operator>(1., cudaq::matrix_operator("custom_op0", {2, 3}));
    op1 = cudaq::product_operator<cudaq::matrix_operator>(1., cudaq::matrix_operator("custom_op1", {2, 0}));
    sum = op0 + op1;
    sum_reverse = op1 + op0;
    difference = op0 - op1;
    difference_reverse = op1 - op0;

    matrices_0 = {
      utils::position_matrix(level_count + 3),
      utils::momentum_matrix(level_count),
      utils::id_matrix(level_count + 1)};
    matrices_1 = {
      utils::id_matrix(level_count + 3),
      utils::create_matrix(level_count),
      utils::number_matrix(level_count + 1)};
    sum_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) + 
                   cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    diff_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) - 
                    cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    diff_reverse_expected = cudaq::kronecker(matrices_1.begin(), matrices_1.end()) - 
                            cudaq::kronecker(matrices_0.begin(), matrices_0.end());

    utils::checkEqual(sum.to_matrix(dimensions), sum_expected);
    utils::checkEqual(sum_reverse.to_matrix(dimensions), sum_expected);
    utils::checkEqual(difference.to_matrix(dimensions), diff_expected);
    utils::checkEqual(difference_reverse.to_matrix(dimensions), diff_reverse_expected);
}
