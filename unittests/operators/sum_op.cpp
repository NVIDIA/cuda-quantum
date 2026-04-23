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

TEST(OperatorExpressions, checkOperatorSumBasics) {

  // testing some utility functions
  {
    cudaq::spin_op previous;
    cudaq::spin_op expected;
    std::vector<std::size_t> all_degrees = {0, 1, 2, 3};
    for (auto id_target : all_degrees) {
      cudaq::spin_op_term op;
      cudaq::spin_op_term expected_term;
      for (std::size_t target : all_degrees) {
        if (target == id_target)
          op *= cudaq::spin_op::i(target);
        else if (target & 2) {
          op *= cudaq::spin_op::z(target);
          expected_term *= cudaq::spin_op::z(target);
        } else if (target & 1) {
          op *= cudaq::spin_op::x(target);
          expected_term *= cudaq::spin_op::x(target);
        } else {
          op *= cudaq::spin_op::y(target);
          expected_term *= cudaq::spin_op::y(target);
        }
      }
      previous += op;
      expected += expected_term;
      auto got = previous;

      ASSERT_NE(got, expected);
      got.canonicalize();
      ASSERT_EQ(got, expected);
      ASSERT_EQ(got.degrees(), expected.degrees());
      ASSERT_EQ(got.to_matrix(), expected.to_matrix());

      auto check_expansion =
          [&got, &all_degrees](const std::set<std::size_t> &want_degrees) {
            auto canon = got;
            auto term_with_missing_degrees = false;
            for (const auto &term : canon)
              if (term.degrees() != all_degrees)
                term_with_missing_degrees = true;
            ASSERT_TRUE(term_with_missing_degrees);
            canon.canonicalize(want_degrees);
            ASSERT_EQ(canon.degrees(), all_degrees);
            for (const auto &term : canon)
              ASSERT_EQ(term.degrees(), all_degrees);
          };

      check_expansion(
          std::set<std::size_t>(all_degrees.begin(), all_degrees.end()));
      if (id_target > 0)
        // for id_target = 0, there is only one term in the sum, which does not
        // act on a degrees
        check_expansion({});
      auto have_degrees = got.degrees();
      ASSERT_ANY_THROW(got.canonicalize(
          std::set<std::size_t>(have_degrees.begin() + 1, have_degrees.end())));
    }
  }
  {
    srand(10);
    for (auto rep = 0; rep < 10; ++rep) {
      auto bit_mask = rand();
      cudaq::sum_op<cudaq::spin_handler> expected;
      std::vector<std::size_t> degrees;
      for (std::size_t d = 1; d <= 10; ++d)
        degrees.push_back(d);

      std::vector<cudaq::product_op<cudaq::spin_handler>> terms;
      for (auto d : degrees) {
        auto coeff = (bit_mask >> d) & 1 ? 1.0 : 0.0;
        auto prod = coeff * cudaq::spin_op::identity(d);
        if (coeff > 0.)
          expected += prod;
        // randomize the order in which we add terms
        auto rnd_idx = (rand() % d);
        terms.insert(terms.begin() + rnd_idx, std::move(prod));
      }

      cudaq::sum_op<cudaq::spin_handler> orig;
      for (auto &&term : terms)
        orig += term;

      ASSERT_EQ(orig.num_terms(), degrees.size());
      ASSERT_EQ(orig.degrees(), degrees);

      std::cout << "original: " << std::endl;
      std::cout << orig.to_string() << std::endl;
      orig.trim();
      std::cout << "trimmed: " << std::endl;
      std::cout << orig.to_string() << std::endl;
      std::cout << "expected: " << std::endl;
      std::cout << expected.to_string() << std::endl;

      ASSERT_NE(orig.num_terms(), degrees.size());
      ASSERT_EQ(orig.num_terms(), expected.num_terms());
      ASSERT_EQ(orig.degrees(), expected.degrees());
      ASSERT_EQ(orig.to_sparse_matrix(), expected.to_sparse_matrix());

      // check that our term map seems accurate
      for (const auto &term : expected)
        orig += term.degrees()[0] * 1.0 * term;
      ASSERT_EQ(orig.num_terms(), expected.num_terms());
      ASSERT_EQ(orig.degrees(), expected.degrees());
      for (const auto &term : orig) {
        auto got_coeff = term.evaluate_coefficient();
        auto want_coeff = std::complex<double>(term.degrees()[0] + 1.);
        ASSERT_EQ(got_coeff.real(), want_coeff.real());
        ASSERT_EQ(got_coeff.imag(), want_coeff.imag());
      }
    }
  }

  std::vector<int> levels = {2, 3, 4};
  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  {// Same degrees of freedom.
   {auto spin0 = cudaq::spin_op::x(5);
  auto spin1 = cudaq::spin_op::z(5);
  auto spin_sum = spin0 + spin1;

  std::vector<std::size_t> want_degrees = {5};
  auto spin_matrix = utils::PauliX_matrix() + utils::PauliZ_matrix();

  ASSERT_TRUE(spin_sum.degrees() == want_degrees);
  ASSERT_EQ(spin_sum.min_degree(), 5);
  ASSERT_EQ(spin_sum.max_degree(), 5);
  utils::checkEqual(spin_matrix, spin_sum.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_op::number(5);
    auto op1 = cudaq::matrix_op::parity(5);

    auto sum = op0 + op1;
    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_EQ(sum.min_degree(), 5);
    ASSERT_EQ(sum.max_degree(), 5);

    auto got_matrix = sum.to_matrix({{5, level_count}});
    auto matrix0 = utils::number_matrix(level_count);
    auto matrix1 = utils::parity_matrix(level_count);
    auto want_matrix = matrix0 + matrix1;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

// Different degrees of freedom.
{
  auto spin0 = cudaq::spin_op::x(0);
  auto spin1 = cudaq::spin_op::z(1);
  auto spin_sum = spin0 + spin1;

  std::vector<std::size_t> want_degrees = {0, 1};
  auto spin_matrix =
      cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
      cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));

  ASSERT_TRUE(spin_sum.degrees() == want_degrees);
  ASSERT_EQ(spin_sum.min_degree(), 0);
  ASSERT_EQ(spin_sum.max_degree(), 1);
  utils::checkEqual(spin_matrix, spin_sum.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_op::number(0);
    auto op1 = cudaq::matrix_op::parity(1);

    auto got = op0 + op1;
    auto got_reverse = op1 + op0;

    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    ASSERT_EQ(got.min_degree(), 0);
    ASSERT_EQ(got.max_degree(), 1);

    auto got_matrix = got.to_matrix({{0, level_count}, {1, level_count}});
    auto got_matrix_reverse =
        got_reverse.to_matrix({{0, level_count}, {1, level_count}});

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
  auto spin0 = cudaq::spin_op::x(0);
  auto spin1 = cudaq::spin_op::z(2);
  auto spin_sum = spin0 + spin1;

  std::vector<std::size_t> want_degrees = {0, 2};
  auto spin_matrix =
      cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
      cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));

  ASSERT_TRUE(spin_sum.degrees() == want_degrees);
  ASSERT_EQ(spin_sum.min_degree(), 0);
  ASSERT_EQ(spin_sum.max_degree(), 2);
  utils::checkEqual(spin_matrix, spin_sum.to_matrix());

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_op::number(0);
    auto op1 = cudaq::matrix_op::parity(2);

    auto got = op0 + op1;
    auto got_reverse = op1 + op0;

    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    ASSERT_EQ(got.min_degree(), 0);
    ASSERT_EQ(got.max_degree(), 2);

    auto got_matrix = got.to_matrix({{0, level_count}, {2, level_count}});
    auto got_matrix_reverse =
        got_reverse.to_matrix({{0, level_count}, {2, level_count}});

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
  auto spin0 = cudaq::spin_op::x(0);
  auto spin1 = cudaq::spin_op::z(2);
  auto spin_sum = spin0 + spin1;

  std::vector<std::size_t> want_degrees = {0, 2};
  auto spin_matrix =
      cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix()) +
      cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}, {2, 2}};

  ASSERT_TRUE(spin_sum.degrees() == want_degrees);
  utils::checkEqual(spin_matrix, spin_sum.to_matrix(dimensions));

  for (auto level_count : levels) {
    auto op0 = cudaq::matrix_op::number(0);
    auto op1 = cudaq::matrix_op::parity(2);

    auto got = op0 + op1;
    auto got_reverse = op1 + op0;

    std::vector<std::size_t> want_degrees = {0, 2};
    ASSERT_TRUE(got.degrees() == want_degrees);
    ASSERT_TRUE(got_reverse.degrees() == want_degrees);

    dimensions = {{0, level_count}, {1, level_count}, {2, level_count}};
    auto got_matrix = got.to_matrix(dimensions);
    auto got_matrix_reverse = got_reverse.to_matrix(dimensions);

    auto identity = utils::id_matrix(level_count);
    auto matrix0 = utils::number_matrix(level_count);
    auto matrix1 = utils::parity_matrix(level_count);
    std::vector<cudaq::complex_matrix> matrices_0 = {identity, matrix0};
    std::vector<cudaq::complex_matrix> matrices_1 = {matrix1, identity};

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
  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  // matrix operator against constant
  {
    auto op = cudaq::matrix_op::parity(0);
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto sum = scalar_op + op;
    auto reverse = op + scalar_op;

    std::vector<std::size_t> want_degrees = {0};
    auto op_matrix = utils::parity_matrix(2);
    auto scalar_matrix = value_0 * utils::id_matrix(2);

    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix({{0, 2}}));
    utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix({{0, 2}}));
  }

  // spin operator against constant
  {
    auto op = cudaq::spin_op::x(0);
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto sum = scalar_op + op;
    auto reverse = op + scalar_op;

    std::vector<std::size_t> want_degrees = {0};
    auto op_matrix = utils::PauliX_matrix();
    auto scalar_matrix = value_0 * utils::id_matrix(2);

    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_matrix + op_matrix, sum.to_matrix());
    utils::checkEqual(scalar_matrix + op_matrix, reverse.to_matrix());
  }

  // matrix operator against constant from lambda
  {
    auto op = cudaq::matrix_op::parity(1);
    auto scalar_op = cudaq::scalar_operator(function);
    auto sum = scalar_op + op;
    auto reverse = op + scalar_op;

    std::vector<std::size_t> want_degrees = {1};
    auto op_matrix = utils::parity_matrix(2);
    auto scalar_matrix =
        scalar_op.evaluate({{"value", 0.3}}) * utils::id_matrix(2);

    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_matrix + op_matrix,
                      sum.to_matrix({{1, 2}}, {{"value", 0.3}}));
    utils::checkEqual(scalar_matrix + op_matrix,
                      reverse.to_matrix({{1, 2}}, {{"value", 0.3}}));
  }

  // spin operator against constant from lambda
  {
    auto op = cudaq::spin_op::x(1);
    auto scalar_op = cudaq::scalar_operator(function);
    auto sum = scalar_op + op;
    auto reverse = op + scalar_op;

    std::vector<std::size_t> want_degrees = {1};
    auto op_matrix = utils::PauliX_matrix();
    auto scalar_matrix =
        scalar_op.evaluate({{"value", 0.3}}) * utils::id_matrix(2);

    ASSERT_TRUE(sum.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);
    utils::checkEqual(scalar_matrix + op_matrix,
                      sum.to_matrix({{1, 2}}, {{"value", 0.3}}));
    utils::checkEqual(scalar_matrix + op_matrix,
                      reverse.to_matrix({{1, 2}}, {{"value", 0.3}}));
  }
}
}

TEST(OperatorExpressions, checkOperatorSumAgainstScalars) {
  int level_count = 3;
  std::complex<double> value = std::complex<double>(0.1, 0.1);
  double double_value = 0.1;

  // `sum_op + double`
  {
    auto original =
        cudaq::matrix_op::momentum(1) + cudaq::matrix_op::position(2);

    auto sum = original + double_value;
    auto reverse = double_value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::position_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `sum_op + std::complex<double>`
  {
    auto original = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `spin sum + std::complex<double>`
  {
    auto original = cudaq::spin_op::x(1) + cudaq::spin_op::y(2);

    auto sum = original + value;
    auto reverse = value + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto matrix1 =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `sum_op + scalar_operator`
  {
    level_count = 2;
    auto original = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto sum = original + cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) + original;

    ASSERT_TRUE(sum.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

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

  // `sum_op - double`
  {
    auto original = cudaq::matrix_op::parity(1) + cudaq::matrix_op::number(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix =
        difference.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

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

  // `spin sum - double`
  {
    auto original = cudaq::spin_op::x(1) + cudaq::spin_op::z(2);

    auto difference = original - double_value;
    auto reverse = double_value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix = difference.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto matrix1 =
        cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);

    auto want_matrix = sum_matrix - scaled_identity;
    auto want_matrix_reverse = scaled_identity - sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `sum_op - std::complex<double>`
  {
    auto original = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto difference = original - value;
    auto reverse = value - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix =
        difference.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

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

  // `sum_op - scalar_operator`
  {
    auto original = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto difference = original - cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) - original;

    ASSERT_TRUE(difference.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto got_matrix =
        difference.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

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

  // `sum_op * double`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum * double_value;
    auto reverse = double_value * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() ==
                  std::complex<double>(double_value));
    }

    for (const auto &term : reverse) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() ==
                  std::complex<double>(double_value));
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `sum_op * std::complex<double>`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum * value;
    auto reverse = value * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    for (const auto &term : reverse) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `sum_op * scalar_operator`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    for (const auto &term : reverse) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});
    auto got_matrix_reverse =
        reverse.to_matrix({{1, level_count}, {2, level_count + 1}});

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
    auto sum = cudaq::spin_op::i(1) + cudaq::spin_op::y(2);

    auto product = sum * cudaq::scalar_operator(value);
    auto reverse = cudaq::scalar_operator(value) * sum;

    ASSERT_TRUE(product.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    for (const auto &term : reverse) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    auto got_matrix = product.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto matrix0 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto matrix1 =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix, got_matrix_reverse);
  }

  // `sum_op / double`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum / double_value;

    ASSERT_TRUE(product.num_terms() == 2);

    auto expected_coeff = std::complex<double>(1. / double_value);
    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity = (1. / double_value) *
                           utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op / std::complex<double>`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum / value;

    ASSERT_TRUE(product.num_terms() == 2);

    auto expected_coeff = std::complex<double>(1. / value);
    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        (1. / value) * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op / scalar_operator`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    auto product = sum / cudaq::scalar_operator(value);

    ASSERT_TRUE(product.num_terms() == 2);

    auto expected_coeff = std::complex<double>(1. / value);
    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix =
        product.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        (1. / value) * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    auto want_matrix_reverse = scaled_identity * sum_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum / scalar_operator`
  {
    auto sum = cudaq::spin_op::i(1) + cudaq::spin_op::y(2);

    auto product = sum / cudaq::scalar_operator(value);

    ASSERT_TRUE(product.num_terms() == 2);

    auto expected_coeff = std::complex<double>(1. / value);
    for (const auto &term : product) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix = product.to_matrix();

    auto matrix0 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto matrix1 =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto scaled_identity = (1. / value) * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op += double`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    sum += double_value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum += double`
  {
    auto sum = cudaq::spin_op::y(1) + cudaq::spin_op::y(2);

    sum += double_value;
    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, 2}, {2, 2}});
    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto matrix1 =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op += std::complex<double>`
  {
    auto sum = cudaq::matrix_op::momentum(1) + cudaq::matrix_op::squeeze(2);

    sum += value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}},
                                    {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::momentum_matrix(level_count));
    auto matrix1 =
        cudaq::kronecker(utils::squeeze_matrix(level_count + 1, value),
                         utils::id_matrix(level_count));
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op += scalar_operator`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::position(2);

    sum += cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count}, {2, level_count + 1}});

    std::vector<cudaq::complex_matrix> matrices_1 = {
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_2 = {
        utils::position_matrix(level_count + 1), utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity =
        value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = matrix0 + matrix1 + scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= double`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    sum -= double_value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::parity_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));

    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity =
        double_value * utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= std::complex<double>`
  {
    auto sum = cudaq::matrix_op::position(1) + cudaq::matrix_op::number(2);

    sum -= value;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::position_matrix(level_count));
    auto matrix1 = cudaq::kronecker(utils::number_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = matrix0 + matrix1 - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= scalar_operator`
  {
    auto sum = cudaq::matrix_op::number(1) + cudaq::matrix_op::identity(2);

    sum -= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count}, {2, level_count + 1}});

    std::vector<cudaq::complex_matrix> matrices_1 = {
        utils::id_matrix(level_count + 1), utils::number_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_2 = {
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity =
        value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) - scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum -= scalar_operator`
  {
    auto sum = cudaq::spin_op::z(1) + cudaq::spin_op::y(2);

    sum -= cudaq::scalar_operator(value);
    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix();

    std::vector<cudaq::complex_matrix> matrices_1 = {utils::id_matrix(2),
                                                     utils::PauliZ_matrix()};
    std::vector<cudaq::complex_matrix> matrices_2 = {utils::PauliY_matrix(),
                                                     utils::id_matrix(2)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity = value * utils::id_matrix(2 * 2);
    auto want_matrix = matrix0 + matrix1 - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= double`
  {
    auto sum = cudaq::matrix_op::squeeze(1) + cudaq::matrix_op::squeeze(2);

    sum *= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() ==
                  std::complex<double>(double_value));
    }

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}},
                                    {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
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

  // `spin sum *= double`
  {
    auto sum = cudaq::spin_op::y(1) + cudaq::spin_op::i(2);

    sum *= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() ==
                  std::complex<double>(double_value));
    }

    auto got_matrix = sum.to_matrix();
    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto matrix1 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto scaled_identity = double_value * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= std::complex<double>`
  {
    auto sum = cudaq::matrix_op::displace(1) + cudaq::matrix_op::parity(2);

    sum *= value;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}},
                                    {{"displacement", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::displace_matrix(level_count, value));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        value * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= scalar_operator`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::momentum(2);

    sum *= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 2);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      ASSERT_TRUE(term.evaluate_coefficient() == value);
    }

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count}, {2, level_count + 1}});

    std::vector<cudaq::complex_matrix> matrices_1 = {
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_2 = {
        utils::momentum_matrix(level_count + 1), utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity =
        value * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op /= double`
  {
    auto sum = cudaq::matrix_op::squeeze(1) + cudaq::matrix_op::squeeze(2);

    sum /= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    auto expected_coeff = std::complex<double>(1. / double_value);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}},
                                    {{"squeezing", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::squeeze_matrix(level_count, value));
    auto matrix1 =
        cudaq::kronecker(utils::squeeze_matrix(level_count + 1, value),
                         utils::id_matrix(level_count));
    auto sum_matrix = matrix0 + matrix1;
    auto scaled_identity = (1. / double_value) *
                           utils::id_matrix((level_count) * (level_count + 1));

    auto want_matrix = sum_matrix * scaled_identity;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `spin sum /= double`
  {
    auto sum = cudaq::spin_op::y(1) + cudaq::spin_op::i(2);

    sum /= double_value;

    ASSERT_TRUE(sum.num_terms() == 2);
    auto expected_coeff = std::complex<double>(1. / double_value);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix = sum.to_matrix();
    auto matrix0 =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto matrix1 = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto scaled_identity = (1. / double_value) * utils::id_matrix(2 * 2);
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op /= std::complex<double>`
  {
    auto sum = cudaq::matrix_op::displace(1) + cudaq::matrix_op::parity(2);

    sum /= value;

    ASSERT_TRUE(sum.num_terms() == 2);
    auto expected_coeff = std::complex<double>(1. / value);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix = sum.to_matrix({{1, level_count}, {2, level_count + 1}},
                                    {{"displacement", value}});

    auto matrix0 = cudaq::kronecker(utils::id_matrix(level_count + 1),
                                    utils::displace_matrix(level_count, value));
    auto matrix1 = cudaq::kronecker(utils::parity_matrix(level_count + 1),
                                    utils::id_matrix(level_count));
    auto scaled_identity =
        (1. / value) * utils::id_matrix((level_count) * (level_count + 1));
    auto want_matrix = (matrix0 + matrix1) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op /= scalar_operator`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::momentum(2);

    sum /= cudaq::scalar_operator(value);

    ASSERT_TRUE(sum.num_terms() == 2);
    auto expected_coeff = std::complex<double>(1. / value);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 1);
      auto coeff = term.evaluate_coefficient();
      EXPECT_NEAR(coeff.real(), expected_coeff.real(), 1e-8);
      EXPECT_NEAR(coeff.imag(), expected_coeff.imag(), 1e-8);
    }

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count}, {2, level_count + 1}});

    std::vector<cudaq::complex_matrix> matrices_1 = {
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_2 = {
        utils::momentum_matrix(level_count + 1), utils::id_matrix(level_count)};
    auto matrix0 = cudaq::kronecker(matrices_1.begin(), matrices_1.end());
    auto matrix1 = cudaq::kronecker(matrices_2.begin(), matrices_2.end());
    auto scaled_identity =
        (1. / value) * utils::id_matrix((level_count + 1) * level_count);

    auto want_matrix = (matrix0 + matrix1) * scaled_identity;
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

  // `sum_op += product_op`
  {
    auto product = cudaq::matrix_op::number(0) * cudaq::matrix_op::number(1);
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    sum += product;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count + 1}, {2, level_count + 2}});
    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 2), utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix + product_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= product_op`
  {
    auto product = cudaq::matrix_op::number(0) * cudaq::matrix_op::number(1);
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    sum -= product;

    ASSERT_TRUE(sum.num_terms() == 3);

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count + 1}, {2, level_count + 2}});
    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 2), utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count)};

    auto product_matrix =
        cudaq::kronecker(matrices_0_0.begin(), matrices_0_0.end()) *
        cudaq::kronecker(matrices_0_1.begin(), matrices_0_1.end());
    auto sum_matrix =
        cudaq::kronecker(matrices_1_0.begin(), matrices_1_0.end()) +
        cudaq::kronecker(matrices_1_1.begin(), matrices_1_1.end());

    auto want_matrix = sum_matrix - product_matrix;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= product_op`
  {
    auto product = cudaq::matrix_op::number(0) * cudaq::matrix_op::number(1);
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);

    sum *= product;

    ASSERT_TRUE(sum.num_terms() == 2);
    for (const auto &term : sum) {
      ASSERT_TRUE(term.num_ops() == 3);
    }

    auto got_matrix = sum.to_matrix(
        {{0, level_count}, {1, level_count + 1}, {2, level_count + 2}});
    std::vector<cudaq::complex_matrix> matrices_0_0 = {
        utils::id_matrix(level_count + 2), utils::id_matrix(level_count + 1),
        utils::number_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_0_1 = {
        utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};

    std::vector<cudaq::complex_matrix> matrices_1_0 = {
        utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    std::vector<cudaq::complex_matrix> matrices_1_1 = {
        utils::parity_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::id_matrix(level_count)};

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

  // `sum_op + sum_op`
  {
    auto sum_0 = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);
    auto sum_1 = cudaq::matrix_op::parity(0) + cudaq::matrix_op::number(1) +
                 cudaq::matrix_op::parity(3);

    auto sum = sum_0 + sum_1;

    ASSERT_TRUE(sum.num_terms() == 5);

    auto got_matrix = sum.to_matrix({{0, level_count},
                                     {1, level_count + 1},
                                     {2, level_count + 2},
                                     {3, level_count + 3}});

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    std::vector<cudaq::complex_matrix> matrices_1_2;

    matrices_0_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    matrices_1_1 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};
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

  // `sum_op - sum_op`
  {
    auto sum_0 = cudaq::matrix_op::parity(1) + cudaq::matrix_op::position(2);
    auto sum_1 = cudaq::matrix_op::parity(0) + cudaq::matrix_op::number(1) +
                 cudaq::matrix_op::momentum(3);

    auto difference = sum_0 - sum_1;

    ASSERT_TRUE(difference.num_terms() == 5);

    auto got_matrix = difference.to_matrix({{0, level_count},
                                            {1, level_count + 1},
                                            {2, level_count + 2},
                                            {3, level_count + 3}});

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    std::vector<cudaq::complex_matrix> matrices_1_2;

    matrices_0_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::position_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    matrices_1_1 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};
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

  // `sum_op * sum_op`
  {
    auto sum_0 = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);
    auto sum_1 = cudaq::matrix_op::parity(0) + cudaq::matrix_op::number(1) +
                 cudaq::matrix_op::parity(3);

    auto sum_product = sum_0 * sum_1;
    auto sum_product_reverse = sum_1 * sum_0;

    ASSERT_TRUE(sum_product.num_terms() == 6);
    ASSERT_TRUE(sum_product_reverse.num_terms() == 6);
    for (const auto &term : sum_product)
      ASSERT_TRUE(term.num_ops() == 2);
    for (const auto &term : sum_product_reverse)
      ASSERT_TRUE(term.num_ops() == 2);

    auto got_matrix = sum_product.to_matrix({{0, level_count},
                                             {1, level_count + 1},
                                             {2, level_count + 2},
                                             {3, level_count + 3}});
    auto got_matrix_reverse =
        sum_product_reverse.to_matrix({{0, level_count},
                                       {1, level_count + 1},
                                       {2, level_count + 2},
                                       {3, level_count + 3}});

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    std::vector<cudaq::complex_matrix> matrices_1_2;

    matrices_0_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    matrices_1_1 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};
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

  // `sum_op *= sum_op`
  {
    auto sum = cudaq::matrix_op::parity(1) + cudaq::matrix_op::parity(2);
    auto sum_1 = cudaq::matrix_op::parity(0) + cudaq::matrix_op::number(1) +
                 cudaq::matrix_op::parity(3);

    sum *= sum_1;

    ASSERT_TRUE(sum.num_terms() == 6);
    for (const auto &term : sum)
      ASSERT_TRUE(term.num_ops() == 2);

    auto got_matrix = sum.to_matrix({{0, level_count},
                                     {1, level_count + 1},
                                     {2, level_count + 2},
                                     {3, level_count + 3}});

    std::vector<cudaq::complex_matrix> matrices_0_0;
    std::vector<cudaq::complex_matrix> matrices_0_1;
    std::vector<cudaq::complex_matrix> matrices_1_0;
    std::vector<cudaq::complex_matrix> matrices_1_1;
    std::vector<cudaq::complex_matrix> matrices_1_2;

    matrices_0_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::parity_matrix(level_count + 1), utils::id_matrix(level_count)};
    matrices_0_1 = {utils::id_matrix(level_count + 3),
                    utils::parity_matrix(level_count + 2),
                    utils::id_matrix(level_count + 1),
                    utils::id_matrix(level_count)};
    matrices_1_0 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::id_matrix(level_count + 1), utils::parity_matrix(level_count)};
    matrices_1_1 = {
        utils::id_matrix(level_count + 3), utils::id_matrix(level_count + 2),
        utils::number_matrix(level_count + 1), utils::id_matrix(level_count)};
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
  cudaq::dimension_map dimensions = {{0, level_count + 1},
                                     {1, level_count + 2},
                                     {2, level_count},
                                     {3, level_count + 3}};

  {
    auto func0 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::momentum_matrix(dimensions[1]),
                                  utils::position_matrix(dimensions[0]));
        };
    auto func1 =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>> &_none) {
          return cudaq::kronecker(utils::parity_matrix(dimensions[1]),
                                  utils::number_matrix(dimensions[0]));
        };
    cudaq::matrix_handler::define("custom_op0", {-1, -1}, func0);
    cudaq::matrix_handler::define("custom_op1", {-1, -1}, func1);
  }

  auto op0 = cudaq::matrix_handler::instantiate("custom_op0", {0, 1});
  auto op1 = cudaq::matrix_handler::instantiate("custom_op1", {1, 2});
  auto sum = op0 + op1;
  auto sum_reverse = op1 + op0;
  auto difference = op0 - op1;
  auto difference_reverse = op1 - op0;

  std::vector<cudaq::complex_matrix> matrices_0 = {
      utils::id_matrix(level_count), utils::momentum_matrix(level_count + 2),
      utils::position_matrix(level_count + 1)};
  std::vector<cudaq::complex_matrix> matrices_1 = {
      utils::parity_matrix(level_count), utils::number_matrix(level_count + 2),
      utils::id_matrix(level_count + 1)};
  auto sum_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) +
                      cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  auto diff_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) -
                       cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  auto diff_reverse_expected =
      cudaq::kronecker(matrices_1.begin(), matrices_1.end()) -
      cudaq::kronecker(matrices_0.begin(), matrices_0.end());

  utils::checkEqual(sum.to_matrix(dimensions), sum_expected);
  utils::checkEqual(sum_reverse.to_matrix(dimensions), sum_expected);
  utils::checkEqual(difference.to_matrix(dimensions), diff_expected);
  utils::checkEqual(difference_reverse.to_matrix(dimensions),
                    diff_reverse_expected);

  op0 = cudaq::matrix_handler::instantiate("custom_op0", {2, 3});
  op1 = cudaq::matrix_handler::instantiate("custom_op1", {0, 2});
  sum = op0 + op1;
  sum_reverse = op1 + op0;
  difference = op0 - op1;
  difference_reverse = op1 - op0;

  matrices_0 = {utils::momentum_matrix(level_count + 3),
                utils::position_matrix(level_count),
                utils::id_matrix(level_count + 1)};
  matrices_1 = {utils::id_matrix(level_count + 3),
                utils::parity_matrix(level_count),
                utils::number_matrix(level_count + 1)};
  sum_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) +
                 cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  diff_expected = cudaq::kronecker(matrices_0.begin(), matrices_0.end()) -
                  cudaq::kronecker(matrices_1.begin(), matrices_1.end());
  diff_reverse_expected =
      cudaq::kronecker(matrices_1.begin(), matrices_1.end()) -
      cudaq::kronecker(matrices_0.begin(), matrices_0.end());

  utils::checkEqual(sum.to_matrix(dimensions), sum_expected);
  utils::checkEqual(sum_reverse.to_matrix(dimensions), sum_expected);
  utils::checkEqual(difference.to_matrix(dimensions), diff_expected);
  utils::checkEqual(difference_reverse.to_matrix(dimensions),
                    diff_reverse_expected);
}

TEST(OperatorExpressions, checkDefaultValue) {
  cudaq::dimension_map dims = {{0, 2}};
  cudaq::complex_matrix empty;
  auto matrix_term = cudaq::matrix_op::identity(0);
  auto boson_term = cudaq::boson_op::number(0);
  auto matrix_sum = cudaq::matrix_op::number(0) + matrix_term;
  auto boson_sum = cudaq::boson_op::number(0) + boson_term;

  auto matrix_default = cudaq::sum_op<cudaq::matrix_handler>();
  auto matrix_empty = cudaq::sum_op<cudaq::matrix_handler>::empty();
  auto boson_empty = cudaq::sum_op<cudaq::boson_handler>::empty();

  // matrix default + matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default + matrix_sum;
    auto res2 = matrix_sum + sum_default;
    utils::checkEqual(res1.to_matrix(dims), matrix_sum.to_matrix(dims));
    utils::checkEqual(res2.to_matrix(dims), matrix_sum.to_matrix(dims));
  }
  // matrix default - matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default - matrix_sum;
    auto res2 = matrix_sum - sum_default;
    utils::checkEqual(res1.to_matrix(dims), (-matrix_sum).to_matrix(dims));
    utils::checkEqual(res2.to_matrix(dims), matrix_sum.to_matrix(dims));
  }
  // matrix default * matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default * matrix_sum;
    auto res2 = matrix_sum * sum_default;
    utils::checkEqual(res1.to_matrix(dims), matrix_sum.to_matrix(dims));
    utils::checkEqual(res2.to_matrix(dims), matrix_sum.to_matrix(dims));
  }
  // matrix default += matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default += matrix_sum;
    auto res = matrix_sum;
    res += matrix_default;
    utils::checkEqual(sum_default.to_matrix(dims), matrix_sum.to_matrix(dims));
    utils::checkEqual(res.to_matrix(dims), matrix_sum.to_matrix(dims));
  }
  // matrix default -= matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default -= matrix_sum;
    auto res = matrix_sum;
    res -= matrix_default;
    utils::checkEqual(sum_default.to_matrix(dims),
                      (-matrix_sum).to_matrix(dims));
    utils::checkEqual(res.to_matrix(dims), matrix_sum.to_matrix(dims));
  }
  // matrix default *= matrix sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default *= matrix_sum;
    auto res = matrix_sum;
    res *= matrix_default;
    utils::checkEqual(sum_default.to_matrix(dims), matrix_sum.to_matrix(dims));
    utils::checkEqual(res.to_matrix(dims), matrix_sum.to_matrix(dims));
  }

  // matrix default + boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default + boson_sum;
    utils::checkEqual(res.to_matrix(dims), boson_sum.to_matrix(dims));
  }
  // matrix default - boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default - boson_sum;
    utils::checkEqual(res.to_matrix(dims), (-boson_sum).to_matrix(dims));
  }
  // matrix default * boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default * boson_sum;
    utils::checkEqual(res.to_matrix(dims), boson_sum.to_matrix(dims));
  }
  // matrix default += boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default += boson_sum;
    utils::checkEqual(sum_default.to_matrix(dims), boson_sum.to_matrix(dims));
  }
  // matrix default -= boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default -= boson_sum;
    utils::checkEqual(sum_default.to_matrix(dims),
                      (-boson_sum).to_matrix(dims));
  }
  // matrix default *= boson sum
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default *= boson_sum;
    utils::checkEqual(sum_default.to_matrix(dims), boson_sum.to_matrix(dims));
  }

  // matrix default + matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default + matrix_term;
    utils::checkEqual(res.to_matrix(dims), matrix_term.to_matrix(dims));
  }
  // matrix default - matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default - matrix_term;
    utils::checkEqual(res.to_matrix(dims), (-matrix_term).to_matrix(dims));
  }
  // matrix default * matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default * matrix_term;
    utils::checkEqual(res.to_matrix(dims), matrix_term.to_matrix(dims));
  }
  // matrix default += matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default += matrix_term;
    utils::checkEqual(sum_default.to_matrix(dims), matrix_term.to_matrix(dims));
  }
  // matrix default -= matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default -= matrix_term;
    utils::checkEqual(sum_default.to_matrix(dims),
                      (-matrix_term).to_matrix(dims));
  }
  // matrix default *= matrix term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default *= matrix_term;
    utils::checkEqual(sum_default.to_matrix(dims), matrix_term.to_matrix(dims));
  }

  // matrix default + boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default + boson_term;
    utils::checkEqual(res.to_matrix(dims), boson_term.to_matrix(dims));
  }
  // matrix default - boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default - boson_term;
    utils::checkEqual(res.to_matrix(dims), (-boson_term).to_matrix(dims));
  }
  // matrix default * boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res = sum_default * boson_term;
    utils::checkEqual(res.to_matrix(dims), boson_term.to_matrix(dims));
  }
  // matrix default += boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default += boson_term;
    utils::checkEqual(sum_default.to_matrix(dims), boson_term.to_matrix(dims));
  }
  // matrix default -= boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default -= boson_term;
    utils::checkEqual(sum_default.to_matrix(dims),
                      (-boson_term).to_matrix(dims));
  }
  // matrix default *= boson term
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    sum_default *= boson_term;
    utils::checkEqual(sum_default.to_matrix(dims), boson_term.to_matrix(dims));
  }

  // matrix default + matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default + matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty + sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }
  // matrix default - matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default - matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty - sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }
  // matrix default * matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default * matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty * sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }
  // matrix default += matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 += matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty;
    res3 += sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }
  // matrix default -= matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 -= matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty;
    res3 -= sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }
  // matrix default *= matrix empty
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 *= matrix_empty;
    auto res2 = res1 * matrix_term;
    auto res3 = matrix_empty;
    res3 *= sum_default;
    auto res4 = res3 * matrix_term;
    utils::checkEqual(res1.to_matrix(dims), empty);
    utils::checkEqual(res2.to_matrix(dims), empty);
    utils::checkEqual(res3.to_matrix(dims), empty);
    utils::checkEqual(res4.to_matrix(dims), empty);
  }

  auto scalar_val = 5.;
  std::complex<double> minus_one = -1.;
  cudaq::complex_matrix scalar_mat(1, 1);
  scalar_mat[{0, 0}] = scalar_val;

  // matrix default + scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default + scalar_val;
    auto res2 = scalar_val + sum_default;
    utils::checkEqual(res1.to_matrix(dims), scalar_mat);
    utils::checkEqual(res2.to_matrix(dims), scalar_mat);
  }
  // matrix default - scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default - scalar_val;
    auto res2 = scalar_val - sum_default;
    utils::checkEqual(res1.to_matrix(dims), minus_one * scalar_mat);
    utils::checkEqual(res2.to_matrix(dims), scalar_mat);
  }
  // matrix default * scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default * scalar_val;
    auto res2 = scalar_val * sum_default;
    utils::checkEqual(res1.to_matrix(dims), scalar_mat);
    utils::checkEqual(res2.to_matrix(dims), scalar_mat);
  }
  // matrix default += scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 += scalar_val;
    utils::checkEqual(res1.to_matrix(dims), scalar_mat);
  }
  // matrix default -= scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 -= scalar_val;
    utils::checkEqual(res1.to_matrix(dims), minus_one * scalar_mat);
  }
  // matrix default *= scalar
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    auto res1 = sum_default;
    res1 *= scalar_val;
    utils::checkEqual(res1.to_matrix(dims), scalar_mat);
  }

  // error cases
  {
    cudaq::sum_op<cudaq::matrix_handler> sum_default;
    utils::checkEqual((+sum_default).to_matrix(dims), empty);

    ASSERT_ANY_THROW(-sum_default);
    ASSERT_ANY_THROW(sum_default / scalar_val);
    ASSERT_ANY_THROW(sum_default /= scalar_val);
  }
}
