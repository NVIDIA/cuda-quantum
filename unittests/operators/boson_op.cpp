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
#include <iostream>

TEST(OperatorExpressions, checkBosonOpsUnary) {
  auto op = cudaq::boson_op::number(0);
  utils::checkEqual((+op).to_matrix({{0, 3}}), utils::number_matrix(3));
  utils::checkEqual((-op).to_matrix({{0, 3}}), -1.0 * utils::number_matrix(3));
  utils::checkEqual(op.to_matrix({{0, 3}}), utils::number_matrix(3));
}

TEST(OperatorExpressions, checkBosonOpsConstruction) {
  auto prod = cudaq::boson_op::identity();
  cudaq::complex_matrix expected(1, 1);

  expected[{0, 0}] = 1.;
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= std::complex<double>(0., -1.);
  expected[{0, 0}] = std::complex<double>(0., -1.);
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= cudaq::boson_op::number(0);
  expected = cudaq::complex_matrix(3, 3);
  expected[{1, 1}] = std::complex<double>(0., -1.);
  expected[{2, 2}] = std::complex<double>(0., -2.);
  utils::checkEqual(prod.to_matrix({{0, 3}}), expected);

  auto sum = cudaq::boson_op::empty();
  expected = cudaq::complex_matrix(0, 0);
  utils::checkEqual(sum.to_matrix(), expected);

  sum *= cudaq::boson_op::number(1); // empty times something is still empty
  std::vector<std::size_t> expected_degrees = {};
  ASSERT_EQ(sum.degrees(), expected_degrees);
  utils::checkEqual(sum.to_matrix(), expected);

  sum += cudaq::boson_op::identity(1);
  expected = cudaq::complex_matrix(3, 3);
  for (size_t i = 0; i < 3; ++i)
    expected[{i, i}] = 1.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum *= cudaq::boson_op::number(1);
  expected = cudaq::complex_matrix(3, 3);
  expected[{1, 1}] = 1.;
  expected[{2, 2}] = 2.;
  utils::checkEqual(sum.to_matrix({{1, 3}}), expected);

  sum = cudaq::boson_op::empty();
  sum -= cudaq::boson_op::identity(0);
  expected = cudaq::complex_matrix(3, 3);
  for (size_t i = 0; i < 3; ++i)
    expected[{i, i}] = -1.;
  utils::checkEqual(sum.to_matrix({{0, 3}}), expected);
}

TEST(OperatorExpressions, checkPreBuiltBosonOps) {

  // number operator
  {
    auto nr_op = cudaq::boson_op::number(0);
    for (auto d = 2; d < 7; ++d) {
      auto nr_mat = utils::number_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = nr_mat;
        auto got = nr_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= nr_mat;
          got *= nr_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }
  }

  // creation operator
  {
    auto ad_op = cudaq::boson_op::create(0);
    for (auto d = 2; d < 7; ++d) {
      auto ad_mat = utils::create_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = ad_mat;
        auto got = ad_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= ad_mat;
          got *= ad_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }
  }

  // annihilation operator
  {
    auto a_op = cudaq::boson_op::annihilate(0);
    for (auto d = 2; d < 7; ++d) {
      auto a_mat = utils::annihilate_matrix(d);
      for (auto pow = 1; pow < 4; ++pow) {
        auto expected = a_mat;
        auto got = a_op;
        for (auto i = 1; i < pow; ++i) {
          expected *= a_mat;
          got *= a_op;
        }
        utils::checkEqual(expected, got.to_matrix({{0, d}}));
      }
    }
  }

  // basic in-place multiplication
  {
    auto max_nr_consecutive = 5;
    auto nr_op = cudaq::boson_op::number(0);
    auto ad_op = cudaq::boson_op::create(0);
    auto a_op = cudaq::boson_op::annihilate(0);
    for (auto d = 2; d < 5; ++d) {

      // we use a larger dimension to compute the correct expected matrices
      // to ensure the expected matrix is no impacted by finite-size errors
      auto nr_mat = utils::number_matrix(d + max_nr_consecutive);
      auto ad_mat = utils::create_matrix(d + max_nr_consecutive);
      auto a_mat = utils::annihilate_matrix(d + max_nr_consecutive);

      for (auto nrs = 0; nrs < max_nr_consecutive; ++nrs) {
        for (auto ads = 0; ads < max_nr_consecutive; ++ads) {
          for (auto as = 0; as < max_nr_consecutive; ++as) {

            // Check Ads * Ns * As

            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# As: " << as << std::endl;

            auto padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            auto expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            auto got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // Check  Ads * As * Ns

            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# As: " << as << ", ";
            std::cout << "# Ns: " << nrs << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // Check Ns * Ads * As

            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# As: " << as << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // check Ns * As * Ads

            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# As: " << as << ", ";
            std::cout << "# Ads: " << ads << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // check As * Ns * Ads

            std::cout << "# As: " << as << ", ";
            std::cout << "# Ns: " << nrs << ", ";
            std::cout << "# Ads: " << ads << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));

            // check As * Ads * Ns

            std::cout << "# As: " << as << ", ";
            std::cout << "# Ads: " << ads << ", ";
            std::cout << "# Ns: " << nrs << std::endl;

            padded = utils::id_matrix(d + max_nr_consecutive);
            for (auto i = 0; i < as; ++i)
              padded *= a_mat;
            for (auto i = 0; i < ads; ++i)
              padded *= ad_mat;
            for (auto i = 0; i < nrs; ++i)
              padded *= nr_mat;
            expected = cudaq::complex_matrix(d, d);
            for (std::size_t i = 0; i < d; i++) {
              for (std::size_t j = 0; j < d; j++)
                expected[{i, j}] = padded[{i, j}];
            }

            got = cudaq::boson_op::identity(0);
            for (auto i = 0; i < as; ++i)
              got *= a_op;
            for (auto i = 0; i < ads; ++i)
              got *= ad_op;
            for (auto i = 0; i < nrs; ++i)
              got *= nr_op;

            utils::checkEqual(expected, got.to_matrix({{0, d}}));
          }
        }
      }
    }
  }
}

TEST(OperatorExpressions, checkBosonOpsWithComplex) {
  std::complex<double> value = std::complex<double>(0.125, 0.125);
  auto dimension = 3;

  // `boson_handler` + `complex<double>`
  {
    auto elementary = cudaq::boson_op::create(0);

    auto sum = value + elementary;
    auto reverse = elementary + value;

    auto got_matrix = sum.to_matrix({{0, dimension}});
    auto got_matrix_reverse = reverse.to_matrix({{0, dimension}});

    auto scaled_identity = value * utils::id_matrix(dimension);
    auto want_matrix = scaled_identity + utils::create_matrix(dimension);
    auto want_matrix_reverse =
        utils::create_matrix(dimension) + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `boson_handler` - `complex<double>`
  {
    auto elementary = cudaq::boson_op::number(0);

    auto difference = value - elementary;
    auto reverse = elementary - value;

    auto got_matrix = difference.to_matrix({{0, dimension}});
    auto got_matrix_reverse = reverse.to_matrix({{0, dimension}});

    auto scaled_identity = value * utils::id_matrix(dimension);
    auto want_matrix = scaled_identity - utils::number_matrix(dimension);
    auto want_matrix_reverse =
        utils::number_matrix(dimension) - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `boson_handler` * `complex<double>`
  {
    auto elementary = cudaq::boson_op::annihilate(0);

    auto product = value * elementary;
    auto reverse = elementary * value;

    auto got_matrix = product.to_matrix({{0, dimension}});
    auto got_matrix_reverse = reverse.to_matrix({{0, dimension}});

    auto scaled_identity = value * utils::id_matrix(dimension);
    auto want_matrix = scaled_identity * utils::annihilate_matrix(dimension);
    auto want_matrix_reverse =
        utils::annihilate_matrix(dimension) * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

TEST(OperatorExpressions, checkBosonOpsWithScalars) {

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  /// Keeping these fixed for these more simple tests.
  int degree_index = 0;
  int dimension = 3;
  double const_scale_factor = 2.0;

  // `boson_handler + scalar_operator`
  {
    auto self = cudaq::boson_op::number(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix = sum.to_matrix({{0, dimension}});
    auto got_reverse_matrix = reverse.to_matrix({{0, dimension}});
    auto want_matrix = utils::number_matrix(dimension) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils::number_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler + scalar_operator`
  {
    auto self = cudaq::boson_op::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix =
        sum.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::annihilate_matrix(dimension) + scaled_identity;
    auto want_reverse_matrix =
        scaled_identity + utils::annihilate_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler - scalar_operator`
  {
    auto self = cudaq::boson_op::identity(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix = sum.to_matrix({{0, dimension}});
    auto got_reverse_matrix = reverse.to_matrix({{0, dimension}});
    auto want_matrix = utils::id_matrix(dimension) - scaled_identity;
    auto want_reverse_matrix = scaled_identity - utils::id_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler - scalar_operator`
  {
    auto self = cudaq::boson_op::create(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix =
        sum.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::create_matrix(dimension) - scaled_identity;
    auto want_reverse_matrix =
        scaled_identity - utils::create_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler * scalar_operator`
  {
    auto self = cudaq::boson_op::number(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<std::size_t> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix = product.to_matrix({{0, dimension}});
    auto got_reverse_matrix = reverse.to_matrix({{0, dimension}});
    auto want_matrix = utils::number_matrix(dimension) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils::number_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler * scalar_operator`
  {
    auto self = cudaq::boson_op::annihilate(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<std::size_t> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(dimension);
    auto got_matrix =
        product.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({{0, dimension}}, {{"value", const_scale_factor}});
    auto want_matrix = utils::annihilate_matrix(dimension) * scaled_identity;
    auto want_reverse_matrix =
        scaled_identity * utils::annihilate_matrix(dimension);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }
}

TEST(OperatorExpressions, checkBosonOpsSimpleArithmetics) {
  cudaq::dimension_map dimensions = {{0, 3}, {1, 2}, {2, 4}};

  // Addition, same DOF.
  {
    auto self = cudaq::boson_op::number(0);
    auto other = cudaq::boson_op::annihilate(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::number_matrix(3) + utils::annihilate_matrix(3);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::boson_op::create(0);
    auto other = cudaq::boson_op::identity(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto matrix_self =
        cudaq::kronecker(utils::id_matrix(2), utils::create_matrix(3));
    auto matrix_other =
        cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(3));
    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = matrix_self + matrix_other;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::boson_op::identity(0);
    auto other = cudaq::boson_op::number(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = utils::id_matrix(3) - utils::number_matrix(3);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::boson_op::annihilate(0);
    auto other = cudaq::boson_op::create(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto annihilate_full =
        cudaq::kronecker(utils::id_matrix(2), utils::annihilate_matrix(3));
    auto create_full =
        cudaq::kronecker(utils::create_matrix(2), utils::id_matrix(3));
    auto got_matrix = sum.to_matrix(dimensions);
    auto want_matrix = annihilate_full - create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::boson_op::create(0);
    auto other = cudaq::boson_op::annihilate(0);

    auto product = self * other;
    ASSERT_TRUE(product.num_ops() == 1);

    std::vector<std::size_t> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix(dimensions);
    auto want_matrix = utils::number_matrix(3);
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::boson_op::position(0);
    auto other = cudaq::boson_op::momentum(1);

    auto result =
        self * other; // nnote that position and momentum are each 2-term sums
    ASSERT_TRUE(result.num_terms() == 4);

    std::vector<std::size_t> want_degrees = {0, 1};
    ASSERT_TRUE(result.degrees() == want_degrees);

    auto annihilate_full =
        cudaq::kronecker(utils::id_matrix(2), utils::position_matrix(3));
    auto create_full =
        cudaq::kronecker(utils::momentum_matrix(2), utils::id_matrix(3));
    auto got_matrix = result.to_matrix(dimensions);
    auto want_matrix = annihilate_full * create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkBosonOpsAdvancedArithmetics) {

  // Keeping this fixed throughout.
  std::complex<double> value = std::complex<double>(0.125, 0.5);
  cudaq::dimension_map dimensions = {{0, 3}, {1, 2}, {2, 4}, {3, 2}};

  // `boson_handler + sum_op`
  {
    auto self = cudaq::boson_op::create(2);
    auto sum_op = cudaq::boson_op::annihilate(2) + cudaq::boson_op::number(1);

    auto got = self + sum_op;
    auto reverse = sum_op + self;

    ASSERT_TRUE(got.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils::create_matrix(4), utils::id_matrix(2));
    auto term_0_full =
        cudaq::kronecker(utils::annihilate_matrix(4), utils::id_matrix(2));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(4), utils::number_matrix(2));

    auto got_matrix = got.to_matrix(dimensions);
    auto got_reverse_matrix = reverse.to_matrix(dimensions);
    auto want_matrix = self_full + term_0_full + term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler - sum_op`
  {
    auto self = cudaq::boson_op::annihilate(0);
    auto sum_op = cudaq::boson_op::create(0) + cudaq::boson_op::identity(1);

    auto got = self - sum_op;
    auto reverse = sum_op - self;

    ASSERT_TRUE(got.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils::id_matrix(2), utils::annihilate_matrix(3));
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::create_matrix(3));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(3));

    auto got_matrix = got.to_matrix(dimensions);
    auto got_reverse_matrix = reverse.to_matrix(dimensions);
    auto want_matrix = self_full - term_0_full - term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `boson_handler * sum_op`
  {
    auto self = cudaq::boson_op::number(0);
    auto sum_op = cudaq::boson_op::create(0) + cudaq::boson_op::number(2);

    auto got = self * sum_op;
    auto reverse = sum_op * self;

    ASSERT_TRUE(got.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    for (const auto &term : got)
      ASSERT_TRUE(term.num_ops() == term.degrees().size());
    for (const auto &term : reverse)
      ASSERT_TRUE(term.num_ops() == term.degrees().size());

    auto self_full =
        cudaq::kronecker(utils::id_matrix(4), utils::number_matrix(3));
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(4), utils::create_matrix(3));
    auto term_1_full =
        cudaq::kronecker(utils::number_matrix(4), utils::id_matrix(3));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = got.to_matrix(dimensions);
    auto got_reverse_matrix = reverse.to_matrix(dimensions);
    auto want_matrix = self_full * sum_full;
    auto want_reverse_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `sum_op += boson_handler`
  {
    auto sum_op = cudaq::boson_op::momentum(0) + cudaq::boson_op::annihilate(2);
    sum_op += cudaq::boson_op::position(0);

    ASSERT_TRUE(sum_op.num_terms() == 3);

    auto term_0_full =
        cudaq::kronecker(utils::annihilate_matrix(4), utils::id_matrix(3));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(4), utils::position_matrix(3));
    auto added_full =
        cudaq::kronecker(utils::id_matrix(4), utils::momentum_matrix(3));

    auto got_matrix = sum_op.to_matrix(dimensions);
    auto want_matrix = term_0_full + term_1_full + added_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op -= boson_handler`
  {
    auto sum_op = cudaq::boson_op::create(0) + cudaq::boson_op::annihilate(1);
    sum_op -= cudaq::boson_op::momentum(0);

    ASSERT_TRUE(sum_op.num_terms() == 3);

    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::create_matrix(3));
    auto term_1_full =
        cudaq::kronecker(utils::annihilate_matrix(2), utils::id_matrix(3));
    auto subtr_full =
        cudaq::kronecker(utils::id_matrix(2), utils::momentum_matrix(3));

    auto got_matrix = sum_op.to_matrix(dimensions);
    auto want_matrix = term_0_full + term_1_full - subtr_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `sum_op *= boson_handler`
  {
    auto sum_op = cudaq::boson_op::momentum(0) + cudaq::boson_op::momentum(1);
    auto self = cudaq::boson_op::position(0);

    sum_op *= self;

    ASSERT_TRUE(sum_op.num_terms() == 8);
    for (const auto &term : sum_op)
      ASSERT_TRUE(term.num_ops() == term.degrees().size());

    // Note that here we need to again expand the matrices for the product
    // computation to ensure that the expected matrix is correct.
    // (naive construction with "the right size" will lead to finite size
    // errors).
    auto padded_term0 = utils::momentum_matrix(5) * utils::position_matrix(5);
    auto term0 = cudaq::complex_matrix(3, 3);
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j)
        term0[{i, j}] = padded_term0[{i, j}];
    }

    auto expected_term0 = cudaq::kronecker(utils::id_matrix(2), term0);
    auto expected_term1 =
        cudaq::kronecker(utils::momentum_matrix(2), utils::position_matrix(3));

    auto got_matrix = sum_op.to_matrix(dimensions);
    auto want_matrix = expected_term0 + expected_term1;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkBosonOpsDegreeVerification) {
  auto op1 = cudaq::boson_op::create(2);
  auto op2 = cudaq::boson_op::annihilate(0);
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}, {2, 3}, {3, 3}};

  ASSERT_ANY_THROW(op1.to_matrix({}));
  ASSERT_ANY_THROW(op1.to_matrix({{1, 2}}));
  ASSERT_ANY_THROW((op1 * op2).to_matrix({{2, 3}}));
  ASSERT_ANY_THROW((op1 + op2).to_matrix({{0, 3}}));
  ASSERT_NO_THROW((op1 * op2).to_matrix(dimensions));
  ASSERT_NO_THROW((op1 + op2).to_matrix(dimensions));
}

TEST(OperatorExpressions, checkCommutationRelations) {

  // Doing some testing for the tests - if the reference matrices do not satisfy
  // the correct relations, then all tests are wrong...

  auto ad_mat = utils::create_matrix(5);
  auto a_mat = utils::annihilate_matrix(5);

  auto padded_commutator = a_mat * ad_mat - ad_mat * a_mat;
  cudaq::complex_matrix commutator_mat(4, 4);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j)
      commutator_mat[{i, j}] = padded_commutator[{i, j}];
  }

  auto padded_aad = a_mat * ad_mat;
  cudaq::complex_matrix aad_mat(4, 4);
  for (size_t i = 0; i < 4; ++i) {
    for (size_t j = 0; j < 4; ++j)
      aad_mat[{i, j}] = padded_aad[{i, j}];
  }

  utils::checkEqual(commutator_mat, utils::id_matrix(4));
  utils::checkEqual(ad_mat * a_mat, utils::number_matrix(5));
  utils::checkEqual(aad_mat, utils::number_matrix(4) + utils::id_matrix(4));

  // Expected commutation relations:
  // [a(k), a†(q)] = δkq
  // [a†(k), a†(q)] = [a(k), a(q)] = 0

  cudaq::dimension_map dimensions = {{0, 4}, {1, 4}};
  auto commutator = [](cudaq::product_op<cudaq::boson_handler> ad,
                       cudaq::product_op<cudaq::boson_handler> a) {
    return a * ad - ad * a;
  };

  // check [a(q), a†(q)] = 1

  auto rel1 =
      commutator(cudaq::boson_op::create(0), cudaq::boson_op::annihilate(0));
  auto rel2 =
      commutator(cudaq::boson_op::create(1), cudaq::boson_op::annihilate(1));
  utils::checkEqual(rel1.to_matrix(dimensions), utils::id_matrix(4));
  utils::checkEqual(rel2.to_matrix(dimensions), utils::id_matrix(4));

  // check [a(k), a†(q)] = 0 for k != q

  auto rel3 =
      commutator(cudaq::boson_op::create(0), cudaq::boson_op::annihilate(1));
  auto rel4 =
      commutator(cudaq::boson_op::create(1), cudaq::boson_op::annihilate(0));
  utils::checkEqual(rel3.to_matrix(dimensions), utils::zero_matrix(16));
  utils::checkEqual(rel4.to_matrix(dimensions), utils::zero_matrix(16));

  // check [a†(q), a†(q)] = 0

  auto rel5 =
      commutator(cudaq::boson_op::create(0), cudaq::boson_op::create(0));
  auto rel6 =
      commutator(cudaq::boson_op::create(1), cudaq::boson_op::create(1));
  utils::checkEqual(rel5.to_matrix(dimensions), utils::zero_matrix(4));
  utils::checkEqual(rel6.to_matrix(dimensions), utils::zero_matrix(4));

  // check [a(q), a(q)] = 0

  auto rel7 = commutator(cudaq::boson_op::annihilate(0),
                         cudaq::boson_op::annihilate(0));
  auto rel8 = commutator(cudaq::boson_op::annihilate(1),
                         cudaq::boson_op::annihilate(1));
  utils::checkEqual(rel7.to_matrix(dimensions), utils::zero_matrix(4));
  utils::checkEqual(rel8.to_matrix(dimensions), utils::zero_matrix(4));

  // check [a†(k), a†(q)] = 0 for k != q

  auto rel9 =
      commutator(cudaq::boson_op::create(0), cudaq::boson_op::create(1));
  auto rel10 =
      commutator(cudaq::boson_op::create(1), cudaq::boson_op::create(0));
  utils::checkEqual(rel9.to_matrix(dimensions), utils::zero_matrix(16));
  utils::checkEqual(rel10.to_matrix(dimensions), utils::zero_matrix(16));

  // check [a(k), a(q)] = 0 for k != q

  auto rel11 = commutator(cudaq::boson_op::annihilate(0),
                          cudaq::boson_op::annihilate(1));
  auto rel12 = commutator(cudaq::boson_op::annihilate(1),
                          cudaq::boson_op::annihilate(0));
  utils::checkEqual(rel11.to_matrix(dimensions), utils::zero_matrix(16));
  utils::checkEqual(rel12.to_matrix(dimensions), utils::zero_matrix(16));
}

TEST(OperatorExpressions, checkMultiDiagConversionBoson) {
  cudaq::dimension_map dimensions = {{0, 10}, {1, 10}};
  // Same degree of freedom
  for (auto &H1 : {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
                   cudaq::boson_op::number(0)}) {
    for (auto &H2 : {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
                     cudaq::boson_op::number(0)}) {
      for (auto &H3 :
           {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
            cudaq::boson_op::number(0)}) {
        for (auto &H4 :
             {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
              cudaq::boson_op::number(0)}) {
          std::cout << H1.to_string() << " " << H2.to_string() << " "
                    << H3.to_string() << " " << H4.to_string() << "\n";
          auto H = H1 * H2 * H3 * H4;
          utils::checkEqual(H.to_matrix(dimensions),
                            H.to_diagonal_matrix(dimensions));
        }
      }
    }
  }

  // Different degrees of freedom
  for (auto &H1 : {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
                   cudaq::boson_op::number(0)}) {
    for (auto &H2 : {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
                     cudaq::boson_op::number(0)}) {
      for (auto &H3 :
           {cudaq::boson_op::annihilate(1), cudaq::boson_op::create(1),
            cudaq::boson_op::number(1)}) {
        for (auto &H4 :
             {cudaq::boson_op::annihilate(1), cudaq::boson_op::create(1),
              cudaq::boson_op::number(1)}) {
          std::cout << H1.to_string() << " " << H2.to_string() << " "
                    << H3.to_string() << " " << H4.to_string() << "\n";
          auto H = H1 * H2 * H3 * H4;
          utils::checkEqual(H.to_matrix(dimensions),
                            H.to_diagonal_matrix(dimensions));
        }
      }
    }
  }

  // Sum op
  for (auto &H1 : {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
                   cudaq::boson_op::number(0)}) {
    for (auto &H2 : {cudaq::boson_op::annihilate(1), cudaq::boson_op::create(1),
                     cudaq::boson_op::number(1)}) {
      for (auto &H3 :
           {cudaq::boson_op::annihilate(0), cudaq::boson_op::create(0),
            cudaq::boson_op::number(0)}) {
        for (auto &H4 :
             {cudaq::boson_op::annihilate(1), cudaq::boson_op::create(1),
              cudaq::boson_op::number(1)}) {

          auto H = H1 * H2 + H3 * H4;
          std::cout << H1.to_string() << " " << H2.to_string() << " + "
                    << H3.to_string() << " " << H4.to_string() << " = "
                    << H.to_string() << "\n";
          utils::checkEqual(H.to_matrix(dimensions),
                            H.to_diagonal_matrix(dimensions));
        }
      }
    }
  }
}
