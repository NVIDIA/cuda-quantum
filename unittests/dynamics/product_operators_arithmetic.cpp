/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/matrix.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>

#include <numeric>

namespace utils_1 {
void checkEqual(cudaq::matrix_2 a, cudaq::matrix_2 b) {
  ASSERT_EQ(a.get_rank(), b.get_rank());
  ASSERT_EQ(a.get_rows(), b.get_rows());
  ASSERT_EQ(a.get_columns(), b.get_columns());
  ASSERT_EQ(a.get_size(), b.get_size());
  for (std::size_t i = 0; i < a.get_rows(); i++) {
    for (std::size_t j = 0; j < a.get_columns(); j++) {
      double a_val = a[{i, j}].real();
      double b_val = b[{i, j}].real();
      EXPECT_NEAR(a_val, b_val, 1e-8);
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

// cudaq::matrix_2 displace_matrix(std::size_t size,
//                                       std::complex<double> amplitude) {
//   auto mat = cudaq::matrix_2(size, size);
//   for (std::size_t i = 0; i + 1 < size; i++) {
//     mat[{i + 1, i}] =
//         amplitude * std::sqrt(static_cast<double>(i + 1)) + 0.0 * 'j';
//     mat[{i, i + 1}] = -1. * std::conj(amplitude) * (0.5 * 'j') *
//                         std::sqrt(static_cast<double>(i + 1)) +
//                     0.0 * 'j';
//   }
//   return mat.exp();
// }

} // namespace utils_1

/// TODO: Not yet testing the output matrices coming from this arithmetic.

TEST(OperatorExpressions, checkProductOperatorSimpleMatrixChecks) {
  std::vector<int> levels = {2, 3, 4};

  {
    // Same degrees of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(0);

        cudaq::product_operator got = op0 * op1;

        // auto got_matrix = got.to_matrix({{0, level_count}}, {});

        // auto matrix0 = _annihilate_matrix(level_count);
        // auto matrix1 = _create_matrix(level_count);
        // auto want_matrix = matrix0 * matrix1;

        // ASSERT_TRUE(want_matrix == got_matrix);

        std::vector<int> want_degrees = {0};
        ASSERT_TRUE(got.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(1);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix =
        //     got.to_matrix({{0, level_count}, {1, level_count}}, {});

        cudaq::product_operator got_reverse = op1 * op0;
        // auto got_matrix_reverse =
        //     got_reverse.to_matrix({{0, level_count}, {1, level_count}}, {});

        // auto identity = _id_matrix(level_count);
        // auto matrix0 = _annihilate_matrix(level_count);
        // auto matrix1 = _create_matrix(level_count);

        // auto fullHilbert0 = identity.kronecker(matrix0);
        // auto fullHilbert1 = matrix1.kronecker(identity);
        // auto want_matrix = fullHilbert0 * fullHilbert1;
        // auto want_matrix_reverse = fullHilbert1 * fullHilbert0;

        // ASSERT_TRUE(want_matrix == got_matrix);
        // ASSERT_TRUE(want_matrix_reverse == got_matrix_reverse);

        std::vector<int> want_degrees = {0, 1};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom, non-consecutive.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix = got.to_matrix({{0,level_count},{2,level_count}},
        // {});

        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }

    // Different degrees of freedom, non-consecutive but all dimensions
    // provided.
    {
      for (auto level_count : levels) {
        auto op0 = cudaq::elementary_operator::annihilate(0);
        auto op1 = cudaq::elementary_operator::create(2);

        cudaq::product_operator got = op0 * op1;
        // auto got_matrix =
        // got.to_matrix({{0,level_count},{1,level_count},{2,level_count}}, {});

        cudaq::product_operator got_reverse = op1 * op0;

        std::vector<int> want_degrees = {0, 2};
        ASSERT_TRUE(got.degrees() == want_degrees);
        ASSERT_TRUE(got_reverse.degrees() == want_degrees);
      }
    }
  }
}

TEST(OperatorExpressions, checkProductOperatorSimpleContinued) {

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  auto local_variable = true;
  auto function = [&](std::map<std::string, std::complex<double>> parameters) {
    if (!local_variable)
      throw std::runtime_error("Local variable not detected.");
    return parameters["value"];
  };

  // Scalar Ops against Elementary Ops
  {
    // Annihilation against constant.
    {
      auto id_op = cudaq::elementary_operator::annihilate(0);
      auto scalar_op = cudaq::scalar_operator(value_0);

      auto got = scalar_op * id_op;
      auto got_reverse = scalar_op * id_op;

      std::vector<int> want_degrees = {0};
      ASSERT_TRUE(got.degrees() == want_degrees);
      ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    }

    // Annihilation against constant from lambda.
    {
      auto id_op = cudaq::elementary_operator::annihilate(1);
      auto scalar_op = cudaq::scalar_operator(function);

      auto got = scalar_op * id_op;
      auto got_reverse = scalar_op * id_op;

      std::vector<int> want_degrees = {1};
      ASSERT_TRUE(got.degrees() == want_degrees);
      ASSERT_TRUE(got_reverse.degrees() == want_degrees);
    }
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstScalars) {
  std::complex<double> value_0 = 0.1 + 0.1;

  /// `product_operator + complex<double>`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto sum = value_0 + product_op;
    auto reverse = product_op + value_0;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(sum.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator + double`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto sum = 2.0 + product_op;
    auto reverse = product_op + 2.0;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(sum.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator + scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);
    auto scalar_op = cudaq::scalar_operator(1.0);

    auto sum = scalar_op + product_op;
    auto reverse = product_op + scalar_op;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(sum.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator - complex<double>`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto difference = value_0 - product_op;
    auto reverse = product_op - value_0;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(difference.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator - double`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    auto difference = 2.0 - product_op;
    auto reverse = product_op - 2.0;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(difference.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator - scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);
    auto scalar_op = cudaq::scalar_operator(1.0);

    auto difference = scalar_op - product_op;
    auto reverse = product_op - scalar_op;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(difference.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator * complex<double>`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    ASSERT_TRUE(product_op.term_count() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate({}) == std::complex<double>(1.));

    auto product = value_0 * product_op;
    auto reverse = product_op * value_0;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == value_0);
    ASSERT_TRUE(reverse.get_coefficient().evaluate({}) == value_0);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator * double`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    ASSERT_TRUE(product_op.term_count() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate({}) == std::complex<double>(1.));

    auto product = 2.0 * product_op;
    auto reverse = product_op * 2.0;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == std::complex<double>(2.));
    ASSERT_TRUE(reverse.get_coefficient().evaluate({}) == std::complex<double>(2.));

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator * scalar_operator`
  {
    auto product_op = cudaq::elementary_operator::annihilate(0) *
                      cudaq::elementary_operator::annihilate(1);

    ASSERT_TRUE(product_op.term_count() == 2);
    ASSERT_TRUE(product_op.get_coefficient().evaluate({}) == std::complex<double>(1.));

    auto scalar_op = cudaq::scalar_operator(0.1);
    auto product = scalar_op * product_op;
    auto reverse = product_op * scalar_op;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == scalar_op.evaluate({}));
    ASSERT_TRUE(reverse.get_coefficient().evaluate({}) == scalar_op.evaluate({}));

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
    // ASSERT_TRUE(reverse.degrees() == want_degrees);
  }

  /// `product_operator *= complex<double>`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    product *= value_0;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == value_0);

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
  }

  /// `product_operator *= double`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    product *= 2.0;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == std::complex<double>(2.));

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
  }

  /// `product_operator *= scalar_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto scalar_op = cudaq::scalar_operator(0.1);
    product *= scalar_op;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(product.get_coefficient().evaluate({}) == scalar_op.evaluate({}));
    ASSERT_TRUE(scalar_op.evaluate({}) == std::complex<double>(0.1));

    std::vector<int> want_degrees = {0, 1};
    // ASSERT_TRUE(product.degrees() == want_degrees);
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstProduct) {

  // `product_operator + product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    auto sum = term_0 + term_1;

    ASSERT_TRUE(sum.term_count() == 2);
  }

  // `product_operator - product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    auto difference = term_0 - term_1;

    ASSERT_TRUE(difference.term_count() == 2);
  }

  // `product_operator * product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    auto product = term_0 * term_1;

    ASSERT_TRUE(product.term_count() == 4);
  }

  // `product_operator *= product_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto term_1 = cudaq::elementary_operator::create(1) *
                  cudaq::elementary_operator::annihilate(2);

    term_0 *= term_1;

    ASSERT_TRUE(term_0.term_count() == 4);
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstElementary) {

  // `product_operator + elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto sum = product + elementary;
    auto reverse = elementary + product;

    ASSERT_TRUE(sum.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

  // `product_operator - elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto difference = product - elementary;
    auto reverse = elementary - product;

    ASSERT_TRUE(difference.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);
  }

  // `product_operator * elementary_operator`
  {
    auto term_0 = cudaq::elementary_operator::annihilate(0) *
                  cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    auto product = term_0 * elementary;
    auto reverse = elementary * term_0;

    ASSERT_TRUE(product.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `product_operator *= elementary_operator`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto elementary = cudaq::elementary_operator::create(1);

    product *= elementary;

    ASSERT_TRUE(product.term_count() == 3);
  }
}

TEST(OperatorExpressions, checkProductOperatorAgainstOperatorSum) {

  // `product_operator + operator_sum`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto original_sum = cudaq::elementary_operator::create(1) +
                        cudaq::elementary_operator::create(2);

    auto sum = product + original_sum;
    auto reverse = original_sum + product;

    ASSERT_TRUE(sum.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `product_operator - operator_sum`
  {
    auto product = cudaq::elementary_operator::annihilate(0) *
                   cudaq::elementary_operator::annihilate(1);
    auto original_sum = cudaq::elementary_operator::create(1) +
                        cudaq::elementary_operator::create(2);

    auto difference = product - original_sum;
    auto reverse = original_sum - product;

    ASSERT_TRUE(difference.term_count() == 3);
    ASSERT_TRUE(reverse.term_count() == 3);
  }

  // `product_operator * operator_sum`
  {
    auto original_product = cudaq::elementary_operator::annihilate(0) *
                            cudaq::elementary_operator::annihilate(1);
    auto sum = cudaq::elementary_operator::create(1) +
               cudaq::elementary_operator::create(2);

    auto product = original_product * sum;
    auto reverse = sum * original_product;

    ASSERT_TRUE(product.term_count() == 2);
    ASSERT_TRUE(reverse.term_count() == 2);

    for (auto term : product.get_terms()) {
      ASSERT_TRUE(term.term_count() == 3);
    }

    for (auto term : reverse.get_terms()) {
      ASSERT_TRUE(term.term_count() == 3);
    }
  }
}
