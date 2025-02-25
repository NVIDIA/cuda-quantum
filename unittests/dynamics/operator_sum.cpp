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
#include "cudaq/dynamics/spin_operators.h"

TEST(OperatorExpressions, checkOperatorSumBasics) {
  std::vector<int> levels = {2, 3, 4};

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  {
    // Same degrees of freedom.
    {
      auto spin0 = cudaq::spin_operator::x(5);
      auto spin1 = cudaq::spin_operator::z(5);
      auto spin_sum = spin0 + spin1;

      std::vector<int> want_degrees = {5};
      auto spin_matrix = utils::PauliX_matrix() + utils::PauliZ_matrix();

      ASSERT_TRUE(spin_sum.degrees() == want_degrees);
      utils::checkEqual(spin_matrix, spin_sum.to_matrix());

      for (auto level_count : levels) {
        auto op0 = cudaq::matrix_operator::number(5);
        auto op1 = cudaq::matrix_operator::parity(5);

        auto sum = op0 + op1;
        ASSERT_TRUE(sum.degrees() == want_degrees);

        auto got_matrix = sum.to_matrix({{5, level_count}});
        auto matrix0 = utils::number_matrix(level_count);
        auto matrix1 = utils::parity_matrix(level_count);
        auto want_matrix = matrix0 + matrix1;
        utils::checkEqual(want_matrix, got_matrix);
      }
    }

    // Different degrees of freedom.
    {
      auto spin0 = cudaq::spin_operator::x(0);
      auto spin1 = cudaq::spin_operator::z(1);
      auto spin_sum = spin0 + spin1;

      std::vector<int> want_degrees = {1, 0};
      auto spin_matrix = cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
                         cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));

      ASSERT_TRUE(spin_sum.degrees() == want_degrees);
      utils::checkEqual(spin_matrix, spin_sum.to_matrix());

      for (auto level_count : levels) {
        auto op0 = cudaq::matrix_operator::number(0);
        auto op1 = cudaq::matrix_operator::parity(1);

        auto got = op0 + op1;
        auto got_reverse = op1 + op0;

        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);

        auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
        auto got_matrix_reverse = got_reverse.to_matrix({{0, level_count}, {1, level_count}});

        auto identity = utils::id_matrix(level_count);
        auto matrix0 = utils::number_matrix(level_count);
        auto matrix1 = utils::parity_matrix(level_count);

        auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
        auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
        auto want_matrix = fullHilbert0 + fullHilbert1;

        utils::checkEqual(want_matrix, got_matrix);
        utils::checkEqual(want_matrix, got_matrix_reverse);
      }
    }

    // Different degrees of freedom, non-consecutive.
    // Should produce the same matrices as the above test.
    {
      auto spin0 = cudaq::spin_operator::x(0);
      auto spin1 = cudaq::spin_operator::z(2);
      auto spin_sum = spin0 + spin1;

      std::vector<int> want_degrees = {2, 0};
      auto spin_matrix = cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
                         cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));

      ASSERT_TRUE(spin_sum.degrees() == want_degrees);
      utils::checkEqual(spin_matrix, spin_sum.to_matrix());

      for (auto level_count : levels) {
        auto op0 = cudaq::matrix_operator::number(0);
        auto op1 = cudaq::matrix_operator::parity(2);

        auto got = op0 + op1;
        auto got_reverse = op1 + op0;

        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);

        auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}});
        auto got_matrix_reverse = got_reverse.to_matrix({{0,level_count},{2,level_count}});

        auto identity = utils::id_matrix(level_count);
        auto matrix0 = utils::number_matrix(level_count);
        auto matrix1 = utils::parity_matrix(level_count);

        auto fullHilbert0 = cudaq::kronecker(identity, matrix0);
        auto fullHilbert1 = cudaq::kronecker(matrix1, identity);
        auto want_matrix = fullHilbert0 + fullHilbert1;

        utils::checkEqual(want_matrix, got_matrix);
        utils::checkEqual(want_matrix, got_matrix_reverse);
      }
    }

    // Different degrees of freedom, non-consecutive but all dimensions
    // provided.
    {
      auto spin0 = cudaq::spin_operator::x(0);
      auto spin1 = cudaq::spin_operator::z(2);
      auto spin_sum = spin0 + spin1;

      std::vector<int> want_degrees = {2, 0};
      auto spin_matrix = cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
                         cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
      std::unordered_map<int, int> dimensions = {{0, 2},{1, 2},{2, 2}};

      ASSERT_TRUE(spin_sum.degrees() == want_degrees);
      utils::checkEqual(spin_matrix, spin_sum.to_matrix(dimensions));

      for (auto level_count : levels) {
        auto op0 = cudaq::matrix_operator::number(0);
        auto op1 = cudaq::matrix_operator::parity(2);

        auto got = op0 + op1;
        auto got_reverse = op1 + op0;

        std::vector<int> want_degrees = {2, 0};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);

        dimensions = {{0, level_count},{1, level_count},{2, level_count}};
        auto got_matrix = got.to_matrix(dimensions);
        auto got_matrix_reverse = got_reverse.to_matrix(dimensions);

        auto identity = utils::id_matrix(level_count);
        auto matrix0 = utils::number_matrix(level_count);
        auto matrix1 = utils::parity_matrix(level_count);
        std::vector<cudaq::matrix_2> matrices_0 = {identity, matrix0};
        std::vector<cudaq::matrix_2> matrices_1 = {matrix1, identity};

        auto fullHilbert0 = cudaq::kronecker(matrices_0.begin(), matrices_0.end());
        auto fullHilbert1 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
        auto want_matrix = fullHilbert0 + fullHilbert1;
        auto want_matrix_reverse = fullHilbert1 + fullHilbert0;

        utils::checkEqual(want_matrix, got_matrix);
        utils::checkEqual(got_matrix, want_matrix);
      }
    }
  }

  // Scalar Ops against Elementary Ops
  {
    auto function = [](const std::unordered_map<std::string, std::complex<double>> &parameters) {
      auto entry = parameters.find("value");
      if (entry == parameters.end())
        throw std::runtime_error("value not defined in parameters");
      return entry->second;
    };

    // matrix operator against constant
    {
      auto op = cudaq::matrix_operator::parity(0);
      auto scalar_op = cudaq::scalar_operator(value_0);
      auto sum = scalar_op + op;
      auto reverse = op + scalar_op;

      std::vector<int> want_degrees = {0};
      auto op_matrix = utils::parity_matrix(2);
      auto scalar_matrix = value_0 * utils::id_matrix(2);

      ASSERT_TRUE(sum.degrees() == want_degrees);
      ASSERT_TRUE(reverse.degrees() == want_degrees);
      utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix({{0, 2}}));
      utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix({{0, 2}}));
    }

    // spin operator against constant
    {
      auto op = cudaq::spin_operator::x(0);
      auto scalar_op = cudaq::scalar_operator(value_0);
      auto sum = scalar_op + op;
      auto reverse = op + scalar_op;

      std::vector<int> want_degrees = {0};
      auto op_matrix = utils::PauliX_matrix();
      auto scalar_matrix = value_0 * utils::id_matrix(2);

      ASSERT_TRUE(sum.degrees() == want_degrees);
      ASSERT_TRUE(reverse.degrees() == want_degrees);
      utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix());
      utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix());
    }

    // matrix operator against constant from lambda
    {
      auto op = cudaq::matrix_operator::parity(1);
      auto scalar_op = cudaq::scalar_operator(function);
      auto sum = scalar_op + op;
      auto reverse = op + scalar_op;

      std::vector<int> want_degrees = {1};
      auto op_matrix = utils::parity_matrix(2);
      auto scalar_matrix = scalar_op.evaluate({{"value", 0.3}}) * utils::id_matrix(2);

      ASSERT_TRUE(sum.degrees() == want_degrees);
      ASSERT_TRUE(reverse.degrees() == want_degrees);
      utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix({{1, 2}}, {{"value", 0.3}}));
      utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix({{1, 2}}, {{"value", 0.3}}));
    }

    // spin operator against constant from lambda
    {
      auto op = cudaq::spin_operator::x(1);
      auto scalar_op = cudaq::scalar_operator(function);
      auto sum = scalar_op + op;
      auto reverse = op + scalar_op;

      std::vector<int> want_degrees = {1};
      auto op_matrix = utils::PauliX_matrix();
      auto scalar_matrix = scalar_op.evaluate({{"value", 0.3}}) * utils::id_matrix(2);

      ASSERT_TRUE(sum.degrees() == want_degrees);
      ASSERT_TRUE(reverse.degrees() == want_degrees);
      utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix({{1, 2}}, {{"value", 0.3}}));
      utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix({{1, 2}}, {{"value", 0.3}}));
    }
  }
}

TEST(OperatorExpressions, checkOperatorSumAgainstScalars) {
  int level_count = 3;
  std::complex<double> value = 0.1 + 0.1j;
  double double_value = 0.1;

  // `operator_sum + double`
  {
    auto original = cudaq::matrix_operator::momentum(1) +
                    cudaq::matrix_operator::position(2);

    auto sum = original + double_value;
    auto reverse = double_value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::position_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `operator_sum + std::complex<double>`
  {
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::parity(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `spin sum + std::complex<double>`
  {
    auto original = cudaq::spin_operator::x(1) +
                    cudaq::spin_operator::y(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 = cudaq::kronecker(utils::id_matrix(2),
                                    utils::PauliX_matrix());
    auto matrix1 = cudaq::kronecker(utils::PauliY_matrix(),
                                    utils::id_matrix(2));
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `operator_sum + scalar_operator`
  {
    level_count = 2;
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::parity(2);

    auto sum = original + cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);


    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2,level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix + scaled_identity;
    auto want_matrix_reverse = scaled_identity + sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - double`
  {
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::number(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix({{1, level_count}, {2, level_count+1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin sum - double`
  {
    auto original = cudaq::spin_operator::x(1) +
                    cudaq::spin_operator::z(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 = cudaq::kronecker(utils::id_matrix(2),
                                    utils::PauliX_matrix());
    auto matrix1 = cudaq::kronecker(utils::PauliZ_matrix(),
                                    utils::id_matrix(2));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - std::complex<double>`
  {
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::parity(2);

    auto difference = original - value;
    auto reverse = value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix({{1,level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum - scalar_operator`
  {
    auto original = cudaq::matrix_operator::parity(1) +
                    cudaq::matrix_operator::parity(2);

    auto difference = original - cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix({{1, level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `operator_sum * double`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    auto product = sum * double_value;
    auto reverse = double_value * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    auto got_matrix = product.to_matrix({{1, level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `operator_sum * std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    auto product = sum * value;
    auto reverse = value * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = product.to_matrix({{1,level_count}, {2, level_count+1}}); 
    auto got_matrix_reverse = reverse.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `operator_sum * scalar_operator`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse = reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin sum * scalar_operator`
  {
    auto sum = cudaq::spin_operator::i(1) +
               cudaq::spin_operator::y(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = product.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 = cudaq::kronecker(utils::id_matrix(2),
                                    utils::id_matrix(2));
    auto matrix1 = cudaq::kronecker(utils::PauliY_matrix(),
                                    utils::id_matrix(2));
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `operator_sum *= double`
  {
    auto sum = cudaq::matrix_operator::squeeze(1) +
               cudaq::matrix_operator::squeeze(2);

    sum *= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
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
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum *= double`
  {
    auto sum = cudaq::spin_operator::y(1) +
               cudaq::spin_operator::i(2);

    sum *= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == std::complex<double>(double_value));
    }

    auto got_matrix = sum.to_matrix();
    auto matrix0 = cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto matrix1 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::displace(1) +
               cudaq::matrix_operator::parity(2);

    sum *= value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}}, {{"displacement", value}});

    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(level_count + 1),
                         utils::displace_matrix(level_count, value));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= scalar_operator`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::momentum(2);

    sum *= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 1);
      ASSERT_TRUE(term.get_coefficient().evaluate() == value);
    }

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}, {2, level_count+1}});

    std::vector<cudaq::matrix_2> matrices_1 = {
        utils::id_matrix(level_count + 1),
        utils::parity_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_2 = {
        utils::momentum_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += double`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    sum += double_value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum += double`
  {
    auto sum = cudaq::spin_operator::y(1) +
               cudaq::spin_operator::y(2);

    sum += double_value;
    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, 2}, {2, 2}});
    auto matrix0 = cudaq::kronecker(utils::id_matrix(2),
                                    utils::PauliY_matrix());
    auto matrix1 = cudaq::kronecker(utils::PauliY_matrix(),
                                    utils::id_matrix(2));
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::momentum(1) +
               cudaq::matrix_operator::squeeze(2);

    sum += value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}}, {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 =
        cudaq::kronecker(utils::squeeze_matrix(level_count + 1, value),
                         utils::id_matrix(level_count));
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum += scalar_operator`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::position(2);

    sum += cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 3);

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

  // `operator_sum -= double`
  {
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    sum -= double_value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= std::complex<double>`
  {
    auto sum = cudaq::matrix_operator::position(1) +
               cudaq::matrix_operator::number(2);

    sum -= value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1,level_count}, {2, level_count+1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::position_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= scalar_operator`
  {
    auto sum = cudaq::matrix_operator::number(1) +
               cudaq::matrix_operator::identity(2);

    sum -= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count}, {2, level_count+1}});

    std::vector<cudaq::matrix_2> matrices_1 = {
        utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_2 = {
        utils::id_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum -= scalar_operator`
  {
    auto sum = cudaq::spin_operator::z(1) +
               cudaq::spin_operator::y(2);

    sum -= cudaq::scalar_operator(value);
    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix();

    std::vector<cudaq::matrix_2> matrices_1 = {utils::id_matrix(2), utils::PauliZ_matrix()};
    std::vector<cudaq::matrix_2> matrices_2 = {utils::PauliY_matrix(), utils::id_matrix(2)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }
}

// FIXME: add tests to check sums against elementary

// FIXME: add tests to combine general sum with spin product
TEST(OperatorExpressions, checkOperatorSumAgainstProduct) {
  // NOTE: Much of the simpler arithmetic between the two is tested in the
  // product operator test file. This mainly just tests the assignment operators
  // between the two types.
  int level_count = 2;

  // `operator_sum += product_operator`
  {
    auto product = cudaq::matrix_operator::number(0) *
                   cudaq::matrix_operator::number(1);
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    sum += product;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
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
    auto product = cudaq::matrix_operator::number(0) *
                   cudaq::matrix_operator::number(1);
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    sum -= product;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
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
    auto product = cudaq::matrix_operator::number(0) *
                   cudaq::matrix_operator::number(1);
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);

    sum *= product;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (auto term : sum.get_terms()) {
      ASSERT_TRUE(term.num_terms() == 3);
    }

    auto got_matrix = sum.to_matrix({{0, level_count}, {1, level_count+1}, {2, level_count+2}});
    std::vector<cudaq::matrix_2> matrices_0_0 = {
        utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1),
        utils::id_matrix(level_count)};

    std::vector<cudaq::matrix_2> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1),
        utils::id_matrix(level_count)};
    std::vector<cudaq::matrix_2> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
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

// FIXME: add tests to combine general sum with spin sum
TEST(OperatorExpressions, checkOperatorSumAgainstOperatorSum) {
  int level_count = 2;

  // `operator_sum + operator_sum`
  {
    auto sum_0 = cudaq::matrix_operator::parity(1) +
                 cudaq::matrix_operator::parity(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::number(1) +
                 cudaq::matrix_operator::parity(3);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.num_terms() == 5);

    auto got_matrix = sum.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::parity_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::number_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::parity_matrix(level_count + 3),
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
    auto sum_0 = cudaq::matrix_operator::parity(1) +
                 cudaq::matrix_operator::position(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::number(1) +
                 cudaq::matrix_operator::momentum(3);

    auto difference = sum_0 - sum_1;

    ASSERT_TRUE(difference.num_terms() == 5);

    auto got_matrix = difference.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::parity_matrix(level_count + 1),
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
                    utils::number_matrix(level_count + 1),
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
    auto sum_0 = cudaq::matrix_operator::parity(1) +
                 cudaq::matrix_operator::parity(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::number(1) +
                 cudaq::matrix_operator::parity(3);

    auto sum_product = sum_0 * sum_1;
    auto sum_product_reverse = sum_1 * sum_0;

    ASSERT_TRUE(sum_product.num_terms() == 6);
    ASSERT_TRUE(sum_product_reverse.num_terms() == 6);
    for (auto term : sum_product.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);
    for (auto term : sum_product_reverse.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);

    auto got_matrix = sum_product.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}}); 
    auto got_matrix_reverse = sum_product_reverse.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::parity_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::number_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::parity_matrix(level_count + 3),
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
    auto sum = cudaq::matrix_operator::parity(1) +
               cudaq::matrix_operator::parity(2);
    auto sum_1 = cudaq::matrix_operator::parity(0) +
                 cudaq::matrix_operator::number(1) +
                 cudaq::matrix_operator::parity(3);

    sum *= sum_1;

    ASSERT_TRUE(sum.num_terms() == 6);
    for (auto term : sum.get_terms())
      ASSERT_TRUE(term.num_terms() == 2);

    auto got_matrix = sum.to_matrix({{0,level_count}, {1, level_count+1}, {2, level_count+2}, {3, level_count+3}});

    std::vector<cudaq::matrix_2> matrices_0_0;
    std::vector<cudaq::matrix_2> matrices_0_1;
    std::vector<cudaq::matrix_2> matrices_1_0;
    std::vector<cudaq::matrix_2> matrices_1_1;
    std::vector<cudaq::matrix_2> matrices_1_2;

    matrices_0_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::parity_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::parity_matrix(level_count)};
    matrices_1_1 = {utils::id_matrix(level_count + 3),
                    utils::id_matrix(level_count + 2),
                    utils::number_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_2 = {utils::parity_matrix(level_count + 3),
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

TEST(OperatorExpressions, checkCustomOperatorSum) {
    auto level_count = 2;
    std::unordered_map<int, int> dimensions = {{0, level_count + 1}, {1, level_count + 2}, {2, level_count}, {3, level_count + 3}};

    {
      auto func0 = [](const std::vector<int> &dimensions,
                      const std::unordered_map<std::string, std::complex<double>> &_none) {
        return cudaq::kronecker(utils::momentum_matrix(dimensions[0]),
                                utils::position_matrix(dimensions[1]));
      };
      auto func1 = [](const std::vector<int> &dimensions,
                      const std::unordered_map<std::string, std::complex<double>> &_none) {
        return cudaq::kronecker(utils::parity_matrix(dimensions[0]),
                                utils::number_matrix(dimensions[1]));
      };
      cudaq::matrix_operator::define("custom_op0", {-1, -1}, func0);
      cudaq::matrix_operator::define("custom_op1", {-1, -1}, func1);
    }

    auto op0 = cudaq::matrix_operator::instantiate("custom_op0", {1, 0});
    auto op1 = cudaq::matrix_operator::instantiate("custom_op1", {2, 1});
    auto sum = op0 + op1;
    auto sum_reverse = op1 + op0;
    auto difference = op0 - op1;
    auto difference_reverse = op1 - op0;

    std::vector<cudaq::matrix_2> matrices_0 = {
      utils::id_matrix(level_count),
      utils::momentum_matrix(level_count + 2),
      utils::position_matrix(level_count + 1)};
    std::vector<cudaq::matrix_2> matrices_1 = {
      utils::parity_matrix(level_count),
      utils::number_matrix(level_count + 2),
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

    op0 = cudaq::matrix_operator::instantiate("custom_op0", {3, 2});
    op1 = cudaq::matrix_operator::instantiate("custom_op1", {2, 0});
    sum = op0 + op1;
    sum_reverse = op1 + op0;
    difference = op0 - op1;
    difference_reverse = op1 - op0;

    matrices_0 = {
      utils::momentum_matrix(level_count + 3),
      utils::position_matrix(level_count),
      utils::id_matrix(level_count + 1)};
    matrices_1 = {
      utils::id_matrix(level_count + 3),
      utils::parity_matrix(level_count),
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
