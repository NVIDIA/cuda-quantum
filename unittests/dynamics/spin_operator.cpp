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

TEST(OperatorExpressions, checkSpinOpsUnary) {
  auto op = cudaq::spin_operator::x(0);
  utils::checkEqual((+op).to_matrix(), utils::PauliX_matrix());
  utils::checkEqual((-op).to_matrix(), -1.0 * utils::PauliX_matrix());
  utils::checkEqual(op.to_matrix(), utils::PauliX_matrix());
}

TEST(OperatorExpressions, checkSpinOpsConstruction) {
  auto prod = cudaq::spin_operator::identity();
  cudaq::complex_matrix expected(1, 1);

  expected[{0, 0}] = 1.;
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= std::complex<double>(0., -1.);
  expected[{0, 0}] = std::complex<double>(0., -1.);
  utils::checkEqual(prod.to_matrix(), expected);

  prod *= cudaq::spin_operator::x(0);
  expected = cudaq::complex_matrix(2, 2);
  expected[{0, 1}] = std::complex<double>(0., -1.);
  expected[{1, 0}] = std::complex<double>(0., -1.);
  utils::checkEqual(prod.to_matrix(), expected);

  auto sum = cudaq::spin_operator::empty();
  expected = cudaq::complex_matrix(0, 0);
  utils::checkEqual(sum.to_matrix(), expected);

  sum *= cudaq::spin_operator::x(1); // empty times something is still empty
  std::vector<int> expected_degrees = {};
  ASSERT_EQ(sum.degrees(), expected_degrees);
  utils::checkEqual(sum.to_matrix(), expected);

  sum += cudaq::spin_operator::i(1);
  expected = cudaq::complex_matrix(2, 2);
  for (size_t i = 0; i < 2; ++i)
    expected[{i, i}] = 1.;
  utils::checkEqual(sum.to_matrix(), expected);

  sum *= cudaq::spin_operator::x(1);
  expected = cudaq::complex_matrix(2, 2);
  expected[{0, 1}] = 1.;
  expected[{1, 0}] = 1.;
  utils::checkEqual(sum.to_matrix(), expected);

  sum = cudaq::spin_operator::empty();
  sum -= cudaq::spin_operator::i(0);
  expected = cudaq::complex_matrix(2, 2);
  for (size_t i = 0; i < 2; ++i)
    expected[{i, i}] = -1.;
  utils::checkEqual(sum.to_matrix(), expected);
}

TEST(OperatorExpressions, checkPreBuiltSpinOps) {

  // Keeping this fixed throughout.
  int degree_index = 0;
  auto id = utils::id_matrix(2);

  // Identity operator.
  {
    auto op = cudaq::spin_operator::i(degree_index);
    auto got = op.to_matrix();
    auto want = utils::id_matrix(2);
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // Z operator.
  {
    auto op = cudaq::spin_operator::z(degree_index);
    auto got = op.to_matrix();
    auto want = utils::PauliZ_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // X operator.
  {
    auto op = cudaq::spin_operator::x(degree_index);
    auto got = op.to_matrix();
    auto want = utils::PauliX_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }

  // Y operator.
  {
    auto op = cudaq::spin_operator::y(degree_index);
    auto got = op.to_matrix();
    auto want = utils::PauliY_matrix();
    utils::checkEqual(want, got);
    utils::checkEqual(id, (op * op).to_matrix());
  }
}

TEST(OperatorExpressions, checkSpinOpsWithComplex) {
  std::complex<double> value = std::complex<double>(0.125, 0.125);

  // `spin_operator` + `complex<double>`
  {
    auto elementary = cudaq::spin_operator::y(0);

    auto sum = value + elementary;
    auto reverse = elementary + value;

    auto got_matrix = sum.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto scaled_identity = value * utils::id_matrix(2);
    auto want_matrix = scaled_identity + utils::PauliY_matrix();
    auto want_matrix_reverse = utils::PauliY_matrix() + scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin_operator` - `complex<double>`
  {
    auto elementary = cudaq::spin_operator::x(0);

    auto difference = value - elementary;
    auto reverse = elementary - value;

    auto got_matrix = difference.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto scaled_identity = value * utils::id_matrix(2);
    auto want_matrix = scaled_identity - utils::PauliX_matrix();
    auto want_matrix_reverse = utils::PauliX_matrix() - scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }

  // `spin_operator` * `complex<double>`
  {
    auto elementary = cudaq::spin_operator::z(0);

    auto product = value * elementary;
    auto reverse = elementary * value;

    auto got_matrix = product.to_matrix();
    auto got_matrix_reverse = reverse.to_matrix();

    auto scaled_identity = value * utils::id_matrix(2);
    auto want_matrix = scaled_identity * utils::PauliZ_matrix();
    auto want_matrix_reverse = utils::PauliZ_matrix() * scaled_identity;

    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_matrix_reverse, got_matrix_reverse);
  }
}

TEST(OperatorExpressions, checkSpinOpsWithScalars) {

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  /// Keeping these fixed for these more simple tests.
  int degree_index = 0;
  double const_scale_factor = 2.0;

  // `spin_operator + scalar_operator`
  {
    auto self = cudaq::spin_operator::x(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = sum.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = utils::PauliX_matrix() + scaled_identity;
    auto want_reverse_matrix = scaled_identity + utils::PauliX_matrix();
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator + scalar_operator`
  {
    auto self = cudaq::spin_operator::y(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self + other;
    auto reverse = other + self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = sum.to_matrix({}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({}, {{"value", const_scale_factor}});
    auto want_matrix = utils::PauliY_matrix() + scaled_identity;
    auto want_reverse_matrix = scaled_identity + utils::PauliY_matrix();
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator - scalar_operator`
  {
    auto self = cudaq::spin_operator::i(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = sum.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = utils::id_matrix(2) - scaled_identity;
    auto want_reverse_matrix = scaled_identity - utils::id_matrix(2);
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator - scalar_operator`
  {
    auto self = cudaq::spin_operator::z(0);
    auto other = cudaq::scalar_operator(function);

    auto sum = self - other;
    auto reverse = other - self;

    ASSERT_TRUE(sum.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = sum.to_matrix({}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({}, {{"value", const_scale_factor}});
    auto want_matrix = utils::PauliZ_matrix() - scaled_identity;
    auto want_reverse_matrix = scaled_identity - utils::PauliZ_matrix();
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator * scalar_operator`
  {
    auto self = cudaq::spin_operator::y(0);
    auto other = cudaq::scalar_operator(const_scale_factor);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = product.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = utils::PauliY_matrix() * scaled_identity;
    auto want_reverse_matrix = scaled_identity * utils::PauliY_matrix();
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator * scalar_operator`
  {
    auto self = cudaq::spin_operator::z(0);
    auto other = cudaq::scalar_operator(function);

    auto product = self * other;
    auto reverse = other * self;

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);
    ASSERT_TRUE(reverse.degrees() == want_degrees);

    auto scaled_identity = const_scale_factor * utils::id_matrix(2);
    auto got_matrix = product.to_matrix({}, {{"value", const_scale_factor}});
    auto got_reverse_matrix =
        reverse.to_matrix({}, {{"value", const_scale_factor}});
    auto want_matrix = utils::PauliZ_matrix() * scaled_identity;
    auto want_reverse_matrix = scaled_identity * utils::PauliZ_matrix();
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }
}

TEST(OperatorExpressions, checkSpinOpsSimpleArithmetics) {

  // Addition, same DOF.
  {
    auto self = cudaq::spin_operator::x(0);
    auto other = cudaq::spin_operator::y(0);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix();
    auto want_matrix = utils::PauliX_matrix() + utils::PauliY_matrix();
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Addition, different DOF's.
  {
    auto self = cudaq::spin_operator::z(0);
    auto other = cudaq::spin_operator::y(1);

    auto sum = self + other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto matrix_self =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliZ_matrix());
    auto matrix_other =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto got_matrix = sum.to_matrix();
    auto want_matrix = matrix_self + matrix_other;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, same DOF.
  {
    auto self = cudaq::spin_operator::z(0);
    auto other = cudaq::spin_operator::x(0);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto got_matrix = sum.to_matrix();
    auto want_matrix = utils::PauliZ_matrix() - utils::PauliX_matrix();
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Subtraction, different DOF's.
  {
    auto self = cudaq::spin_operator::y(0);
    auto other = cudaq::spin_operator::x(1);

    auto sum = self - other;
    ASSERT_TRUE(sum.num_terms() == 2);

    auto annihilate_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto create_full =
        cudaq::kronecker(utils::PauliX_matrix(), utils::id_matrix(2));
    auto got_matrix = sum.to_matrix();
    auto want_matrix = annihilate_full - create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, same DOF.
  {
    auto self = cudaq::spin_operator::y(0);
    auto other = cudaq::spin_operator::z(0);

    auto product = self * other;
    ASSERT_TRUE(product.num_terms() == 1);

    std::vector<int> want_degrees = {0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto got_matrix = product.to_matrix();
    auto want_matrix = utils::PauliY_matrix() * utils::PauliZ_matrix();
    utils::checkEqual(want_matrix, got_matrix);
  }

  // Multiplication, different DOF's.
  {
    auto self = cudaq::spin_operator::x(0);
    auto other = cudaq::spin_operator::z(1);

    auto product = self * other;
    ASSERT_TRUE(product.num_terms() == 2);

    std::vector<int> want_degrees = {1, 0};
    ASSERT_TRUE(product.degrees() == want_degrees);

    auto annihilate_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto create_full =
        cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
    auto got_matrix = product.to_matrix();
    auto want_matrix = annihilate_full * create_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkSpinOpsAdvancedArithmetics) {

  // Keeping this fixed throughout.
  std::complex<double> value = std::complex<double>(0.125, 0.5);

  // `spin_operator + operator_sum`
  {
    auto self = cudaq::spin_operator::y(2);
    auto operator_sum = cudaq::spin_operator::y(2) + cudaq::spin_operator::x(1);

    auto got = self + operator_sum;
    auto reverse = operator_sum + self;

    ASSERT_TRUE(got.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);

    auto self_full =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto term_0_full =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());

    auto got_matrix = got.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = self_full + term_0_full + term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator - operator_sum`
  {
    auto self = cudaq::spin_operator::i(0);
    auto operator_sum = cudaq::spin_operator::x(0) + cudaq::spin_operator::z(1);

    auto got = self - operator_sum;
    auto reverse = operator_sum - self;

    ASSERT_TRUE(got.num_terms() == 3);
    ASSERT_TRUE(reverse.num_terms() == 3);

    auto self_full = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto term_1_full =
        cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));

    auto got_matrix = got.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = self_full - term_0_full - term_1_full;
    auto want_reverse_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `spin_operator * operator_sum`
  {
    auto self = cudaq::spin_operator::y(0);
    auto operator_sum = cudaq::spin_operator::x(0) + cudaq::spin_operator::y(2);

    auto got = self * operator_sum;
    auto reverse = operator_sum * self;

    ASSERT_TRUE(got.num_terms() == 2);
    ASSERT_TRUE(reverse.num_terms() == 2);
    for (auto &term : got.get_terms())
      ASSERT_TRUE(term.num_terms() == term.degrees().size());
    for (auto &term : reverse.get_terms())
      ASSERT_TRUE(term.num_terms() == term.degrees().size());

    auto self_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto term_1_full =
        cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = got.to_matrix();
    auto got_reverse_matrix = reverse.to_matrix();
    auto want_matrix = self_full * sum_full;
    auto want_reverse_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
    utils::checkEqual(want_reverse_matrix, got_reverse_matrix);
  }

  // `operator_sum += spin_operator`
  {
    auto operator_sum = cudaq::spin_operator::z(0) + cudaq::spin_operator::x(2);
    operator_sum += cudaq::spin_operator::y(0);

    ASSERT_TRUE(operator_sum.num_terms() == 3);

    auto self_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliZ_matrix());
    auto term_0_full =
        cudaq::kronecker(utils::PauliX_matrix(), utils::id_matrix(2));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());

    auto got_matrix = operator_sum.to_matrix();
    auto want_matrix = term_0_full + term_1_full + self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum -= spin_operator`
  {
    auto operator_sum = cudaq::spin_operator::x(0) + cudaq::spin_operator::i(1);
    operator_sum -= cudaq::spin_operator::x(0);

    ASSERT_TRUE(operator_sum.num_terms() == 2);

    auto self_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto term_1_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());

    auto got_matrix = operator_sum.to_matrix();
    auto want_matrix = term_0_full + term_1_full - self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }

  // `operator_sum *= spin_operator`
  {
    auto self = cudaq::spin_operator::i(0);
    auto operator_sum = cudaq::spin_operator::y(0) + cudaq::spin_operator::z(1);

    operator_sum *= self;

    ASSERT_TRUE(operator_sum.num_terms() == 2);
    for (auto &term : operator_sum.get_terms())
      ASSERT_TRUE(term.num_terms() == term.degrees().size());

    auto self_full = cudaq::kronecker(utils::id_matrix(2), utils::id_matrix(2));
    auto term_0_full =
        cudaq::kronecker(utils::id_matrix(2), utils::PauliY_matrix());
    auto term_1_full =
        cudaq::kronecker(utils::PauliZ_matrix(), utils::id_matrix(2));
    auto sum_full = term_0_full + term_1_full;

    auto got_matrix = operator_sum.to_matrix();
    auto want_matrix = sum_full * self_full;
    utils::checkEqual(want_matrix, got_matrix);
  }
}

TEST(OperatorExpressions, checkSpinOpsDegreeVerification) {
  auto op1 = cudaq::spin_operator::z(1);
  auto op2 = cudaq::spin_operator::x(0);
  std::map<int, int> dimensions = {{0, 1}, {1, 3}};

  ASSERT_ANY_THROW(op1.to_matrix({{1, 3}}));
  ASSERT_ANY_THROW((op1 * op2).to_matrix({{0, 3}, {1, 3}}));
  ASSERT_ANY_THROW((op1 + op2).to_matrix({{0, 3}}));
  ASSERT_NO_THROW(op1.to_matrix({{0, 3}}));
}