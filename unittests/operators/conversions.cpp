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

TEST(OperatorExpressions, checkElementaryOpsConversions) {

  std::unordered_map<std::string, std::complex<double>> parameters = {
      {"squeezing", 0.5}, {"displacement", 0.25}};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};

  auto matrix_elementary = cudaq::matrix_op::parity(1);
  auto matrix_elementary_expected = utils::parity_matrix(2);
  auto spin_elementary = cudaq::spin_op::y(1);
  auto spin_elementary_expected = utils::PauliY_matrix();
  auto boson_elementary = cudaq::boson_op::annihilate(1);
  auto boson_elementary_expected = utils::annihilate_matrix(2);

  auto checkSumEquals = [dimensions,
                         parameters](cudaq::sum_op<cudaq::matrix_handler> sum,
                                     cudaq::complex_matrix expected,
                                     int expected_num_terms = 2) {
    auto got = sum.to_matrix(dimensions, parameters);
    ASSERT_TRUE(sum.num_terms() == expected_num_terms);
    utils::checkEqual(got, expected);
  };

  auto checkProductEquals =
      [dimensions, parameters](cudaq::product_op<cudaq::matrix_handler> prod,
                               cudaq::complex_matrix expected,
                               int expected_num_terms = 2) {
        auto got = prod.to_matrix(dimensions, parameters);
        ASSERT_TRUE(prod.num_ops() == expected_num_terms);
        utils::checkEqual(got, expected);
      };

  // `elementary + elementary`
  {
    checkSumEquals(matrix_elementary + matrix_elementary,
                   matrix_elementary_expected + matrix_elementary_expected, 1);
    checkSumEquals(spin_elementary + spin_elementary,
                   spin_elementary_expected + spin_elementary_expected, 1);
    checkSumEquals(boson_elementary + boson_elementary,
                   boson_elementary_expected + boson_elementary_expected, 1);
    checkSumEquals(matrix_elementary + spin_elementary,
                   matrix_elementary_expected + spin_elementary_expected);
    checkSumEquals(spin_elementary + matrix_elementary,
                   matrix_elementary_expected + spin_elementary_expected);
    checkSumEquals(matrix_elementary + boson_elementary,
                   matrix_elementary_expected + boson_elementary_expected);
    checkSumEquals(boson_elementary + matrix_elementary,
                   matrix_elementary_expected + boson_elementary_expected);
    checkSumEquals(spin_elementary + boson_elementary,
                   spin_elementary_expected + boson_elementary_expected);
    checkSumEquals(boson_elementary + spin_elementary,
                   spin_elementary_expected + boson_elementary_expected);
  }

  // `elementary - elementary`
  {
    checkSumEquals(matrix_elementary - matrix_elementary,
                   matrix_elementary_expected - matrix_elementary_expected, 1);
    checkSumEquals(spin_elementary - spin_elementary,
                   spin_elementary_expected - spin_elementary_expected, 1);
    checkSumEquals(boson_elementary - boson_elementary,
                   boson_elementary_expected - boson_elementary_expected, 1);
    checkSumEquals(matrix_elementary - spin_elementary,
                   matrix_elementary_expected - spin_elementary_expected);
    checkSumEquals(spin_elementary - matrix_elementary,
                   spin_elementary_expected - matrix_elementary_expected);
    checkSumEquals(matrix_elementary - boson_elementary,
                   matrix_elementary_expected - boson_elementary_expected);
    checkSumEquals(boson_elementary - matrix_elementary,
                   boson_elementary_expected - matrix_elementary_expected);
    checkSumEquals(spin_elementary - boson_elementary,
                   spin_elementary_expected - boson_elementary_expected);
    checkSumEquals(boson_elementary - spin_elementary,
                   boson_elementary_expected - spin_elementary_expected);
  }

  // `elementary * elementary`
  {
    checkProductEquals(matrix_elementary * matrix_elementary,
                       matrix_elementary_expected * matrix_elementary_expected);
    checkProductEquals(spin_elementary * spin_elementary,
                       spin_elementary_expected * spin_elementary_expected, 1);
    checkProductEquals(boson_elementary * boson_elementary,
                       boson_elementary_expected * boson_elementary_expected,
                       1);
    checkProductEquals(matrix_elementary * spin_elementary,
                       matrix_elementary_expected * spin_elementary_expected);
    checkProductEquals(spin_elementary * matrix_elementary,
                       spin_elementary_expected * matrix_elementary_expected);
    checkProductEquals(matrix_elementary * boson_elementary,
                       matrix_elementary_expected * boson_elementary_expected);
    checkProductEquals(boson_elementary * matrix_elementary,
                       boson_elementary_expected * matrix_elementary_expected);
    checkProductEquals(spin_elementary * boson_elementary,
                       spin_elementary_expected * boson_elementary_expected);
    checkProductEquals(boson_elementary * spin_elementary,
                       boson_elementary_expected * spin_elementary_expected);
  }

  // `elementary *= elementary`
  {
    auto matrix_product = cudaq::product_op(matrix_elementary);
    matrix_product *= matrix_elementary;
    checkProductEquals(matrix_product,
                       matrix_elementary_expected * matrix_elementary_expected);

    auto spin_product = cudaq::product_op(spin_elementary);
    spin_product *= spin_elementary;
    checkProductEquals(spin_product,
                       spin_elementary_expected * spin_elementary_expected, 1);

    auto boson_product = cudaq::product_op(boson_elementary);
    boson_product *= boson_elementary;
    checkProductEquals(boson_product,
                       boson_elementary_expected * boson_elementary_expected,
                       1);

    matrix_product = cudaq::product_op(matrix_elementary);
    matrix_product *= spin_elementary;
    checkProductEquals(matrix_product,
                       matrix_elementary_expected * spin_elementary_expected);

    matrix_product = cudaq::product_op(matrix_elementary);
    matrix_product *= boson_elementary;
    checkProductEquals(matrix_product,
                       matrix_elementary_expected * boson_elementary_expected);
  }
}

TEST(OperatorExpressions, checkProductOperatorConversions) {

  std::unordered_map<std::string, std::complex<double>> parameters = {
      {"squeezing", 0.5}, {"displacement", 0.25}};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};
  auto matrix_product =
      cudaq::matrix_op::squeeze(0) * cudaq::matrix_op::displace(1);
  auto matrix_product_expected = cudaq::kronecker(
      utils::displace_matrix(2, 0.25), utils::squeeze_matrix(2, 0.5));
  auto spin_product = cudaq::spin_op::y(1) * cudaq::spin_op::x(0);
  auto spin_product_expected =
      cudaq::kronecker(utils::PauliY_matrix(), utils::PauliX_matrix());
  auto boson_product =
      cudaq::boson_op::annihilate(1) * cudaq::boson_op::number(0);
  auto boson_product_expected =
      cudaq::kronecker(utils::annihilate_matrix(2), utils::number_matrix(2));

  auto checkSumEquals = [dimensions,
                         parameters](cudaq::sum_op<cudaq::matrix_handler> sum,
                                     cudaq::complex_matrix expected,
                                     int expected_num_terms = 2) {
    auto got = sum.to_matrix(dimensions, parameters);
    ASSERT_TRUE(sum.num_terms() == expected_num_terms);
    utils::checkEqual(got, expected);
  };

  auto checkProductEquals =
      [dimensions, parameters](cudaq::product_op<cudaq::matrix_handler> prod,
                               cudaq::complex_matrix expected,
                               int expected_num_terms = 4) {
        auto got = prod.to_matrix(dimensions, parameters);
        ASSERT_TRUE(prod.num_ops() == expected_num_terms);
        utils::checkEqual(got, expected);
      };

  // `product + product`
  {
    checkSumEquals(matrix_product + matrix_product,
                   matrix_product_expected + matrix_product_expected, 1);
    checkSumEquals(spin_product + spin_product,
                   spin_product_expected + spin_product_expected, 1);
    checkSumEquals(boson_product + boson_product,
                   boson_product_expected + boson_product_expected, 1);
    checkSumEquals(matrix_product + spin_product,
                   matrix_product_expected + spin_product_expected);
    checkSumEquals(spin_product + matrix_product,
                   matrix_product_expected + spin_product_expected);
    checkSumEquals(matrix_product + boson_product,
                   matrix_product_expected + boson_product_expected);
    checkSumEquals(boson_product + matrix_product,
                   matrix_product_expected + boson_product_expected);
    checkSumEquals(spin_product + boson_product,
                   spin_product_expected + boson_product_expected);
    checkSumEquals(boson_product + spin_product,
                   spin_product_expected + boson_product_expected);
  }

  // `product - product`
  {
    checkSumEquals(matrix_product - matrix_product,
                   matrix_product_expected - matrix_product_expected, 1);
    checkSumEquals(spin_product - spin_product,
                   spin_product_expected - spin_product_expected, 1);
    checkSumEquals(boson_product - boson_product,
                   boson_product_expected - boson_product_expected, 1);
    checkSumEquals(matrix_product - spin_product,
                   matrix_product_expected - spin_product_expected);
    checkSumEquals(spin_product - matrix_product,
                   spin_product_expected - matrix_product_expected);
    checkSumEquals(matrix_product - boson_product,
                   matrix_product_expected - boson_product_expected);
    checkSumEquals(boson_product - matrix_product,
                   boson_product_expected - matrix_product_expected);
    checkSumEquals(spin_product - boson_product,
                   spin_product_expected - boson_product_expected);
    checkSumEquals(boson_product - spin_product,
                   boson_product_expected - spin_product_expected);
  }

  // `product * product`
  {
    checkProductEquals(matrix_product * matrix_product,
                       matrix_product_expected * matrix_product_expected);
    checkProductEquals(spin_product * spin_product,
                       spin_product_expected * spin_product_expected, 2);
    checkProductEquals(boson_product * boson_product,
                       boson_product_expected * boson_product_expected, 2);
    checkProductEquals(matrix_product * spin_product,
                       matrix_product_expected * spin_product_expected);
    checkProductEquals(spin_product * matrix_product,
                       spin_product_expected * matrix_product_expected);
    checkProductEquals(matrix_product * boson_product,
                       matrix_product_expected * boson_product_expected);
    checkProductEquals(boson_product * matrix_product,
                       boson_product_expected * matrix_product_expected);
    checkProductEquals(spin_product * boson_product,
                       spin_product_expected * boson_product_expected);
    checkProductEquals(boson_product * spin_product,
                       boson_product_expected * spin_product_expected);
  }

  // `product *= product`
  {
    auto matrix_product_0 = matrix_product;
    matrix_product_0 *= matrix_product;
    checkProductEquals(matrix_product_0,
                       matrix_product_expected * matrix_product_expected);

    auto spin_product_0 = spin_product;
    spin_product_0 *= spin_product;
    checkProductEquals(spin_product_0,
                       spin_product_expected * spin_product_expected, 2);

    auto boson_product_0 = boson_product;
    boson_product_0 *= boson_product;
    checkProductEquals(boson_product_0,
                       boson_product_expected * boson_product_expected, 2);

    matrix_product_0 = matrix_product;
    matrix_product_0 *= spin_product;
    checkProductEquals(matrix_product_0,
                       matrix_product_expected * spin_product_expected);

    matrix_product_0 = matrix_product;
    matrix_product_0 *= boson_product;
    checkProductEquals(matrix_product_0,
                       matrix_product_expected * boson_product_expected);
  }
}

TEST(OperatorExpressions, checkOperatorSumConversions) {

  std::unordered_map<std::string, std::complex<double>> parameters = {
      {"squeezing", 0.5}, {"displacement", 0.25}};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};

  auto matrix_product =
      cudaq::matrix_op::squeeze(0) * cudaq::matrix_op::displace(1);
  auto matrix_product_expected = cudaq::kronecker(
      utils::displace_matrix(2, 0.25), utils::squeeze_matrix(2, 0.5));
  auto spin_product = cudaq::spin_op::y(1) * cudaq::spin_op::x(0);
  auto spin_product_expected =
      cudaq::kronecker(utils::PauliY_matrix(), utils::PauliX_matrix());
  auto boson_product =
      cudaq::boson_op::annihilate(1) * cudaq::boson_op::number(0);
  auto boson_product_expected =
      cudaq::kronecker(utils::annihilate_matrix(2), utils::number_matrix(2));

  auto matrix_sum =
      cudaq::matrix_op::squeeze(0) + cudaq::matrix_op::displace(1);
  auto matrix_sum_expected =
      cudaq::kronecker(utils::displace_matrix(2, 0.25), utils::id_matrix(2)) +
      cudaq::kronecker(utils::id_matrix(2), utils::squeeze_matrix(2, 0.5));
  auto spin_sum = cudaq::spin_op::y(1) + cudaq::spin_op::x(0);
  auto spin_sum_expected =
      cudaq::kronecker(utils::PauliY_matrix(), utils::id_matrix(2)) +
      cudaq::kronecker(utils::id_matrix(2), utils::PauliX_matrix());
  auto boson_sum = cudaq::boson_op::annihilate(1) + cudaq::boson_op::number(0);
  auto boson_sum_expected =
      cudaq::kronecker(utils::annihilate_matrix(2), utils::id_matrix(2)) +
      cudaq::kronecker(utils::id_matrix(2), utils::number_matrix(2));

  auto checkSumEquals = [dimensions, parameters](
                            cudaq::sum_op<cudaq::matrix_handler> sum,
                            cudaq::complex_matrix expected, int num_terms = 4) {
    auto got = sum.to_matrix(dimensions, parameters);
    ASSERT_TRUE(sum.num_terms() == num_terms);
    utils::checkEqual(got, expected);
  };

  // `sum + product`
  {
    checkSumEquals(matrix_sum + matrix_product,
                   matrix_sum_expected + matrix_product_expected, 3);
    checkSumEquals(spin_sum + spin_product,
                   spin_sum_expected + spin_product_expected, 3);
    checkSumEquals(boson_sum + boson_product,
                   boson_sum_expected + boson_product_expected, 3);
    checkSumEquals(matrix_sum + spin_product,
                   matrix_sum_expected + spin_product_expected, 3);
    checkSumEquals(spin_sum + matrix_product,
                   spin_sum_expected + matrix_product_expected, 3);
    checkSumEquals(matrix_sum + boson_product,
                   matrix_sum_expected + boson_product_expected, 3);
    checkSumEquals(boson_sum + matrix_product,
                   boson_sum_expected + matrix_product_expected, 3);
    checkSumEquals(spin_sum + boson_product,
                   spin_sum_expected + boson_product_expected, 3);
    checkSumEquals(boson_sum + spin_product,
                   boson_sum_expected + spin_product_expected, 3);
  }

  // `product + sum`
  {
    checkSumEquals(matrix_product + matrix_sum,
                   matrix_product_expected + matrix_sum_expected, 3);
    checkSumEquals(spin_product + spin_sum,
                   spin_product_expected + spin_sum_expected, 3);
    checkSumEquals(boson_product + boson_sum,
                   boson_product_expected + boson_sum_expected, 3);
    checkSumEquals(matrix_product + spin_sum,
                   matrix_product_expected + spin_sum_expected, 3);
    checkSumEquals(spin_product + matrix_sum,
                   spin_product_expected + matrix_sum_expected, 3);
    checkSumEquals(matrix_product + boson_sum,
                   matrix_product_expected + boson_sum_expected, 3);
    checkSumEquals(boson_product + matrix_sum,
                   boson_product_expected + matrix_sum_expected, 3);
    checkSumEquals(spin_product + boson_sum,
                   spin_product_expected + boson_sum_expected, 3);
    checkSumEquals(boson_product + spin_sum,
                   boson_product_expected + spin_sum_expected, 3);
  }

  // `sum + sum`
  {
    checkSumEquals(matrix_sum + matrix_sum,
                   matrix_sum_expected + matrix_sum_expected, 2);
    checkSumEquals(spin_sum + spin_sum, spin_sum_expected + spin_sum_expected,
                   2);
    checkSumEquals(boson_sum + boson_sum,
                   boson_sum_expected + boson_sum_expected, 2);
    checkSumEquals(matrix_sum + spin_sum,
                   matrix_sum_expected + spin_sum_expected);
    checkSumEquals(spin_sum + matrix_sum,
                   matrix_sum_expected + spin_sum_expected);
    checkSumEquals(matrix_sum + boson_sum,
                   matrix_sum_expected + boson_sum_expected);
    checkSumEquals(boson_sum + matrix_sum,
                   matrix_sum_expected + boson_sum_expected);
    checkSumEquals(spin_sum + boson_sum,
                   spin_sum_expected + boson_sum_expected);
    checkSumEquals(boson_sum + spin_sum,
                   spin_sum_expected + boson_sum_expected);
  }

  // `sum - product`
  {
    checkSumEquals(matrix_sum - matrix_product,
                   matrix_sum_expected - matrix_product_expected, 3);
    checkSumEquals(spin_sum - spin_product,
                   spin_sum_expected - spin_product_expected, 3);
    checkSumEquals(boson_sum - boson_product,
                   boson_sum_expected - boson_product_expected, 3);
    checkSumEquals(matrix_sum - spin_product,
                   matrix_sum_expected - spin_product_expected, 3);
    checkSumEquals(spin_sum - matrix_product,
                   spin_sum_expected - matrix_product_expected, 3);
    checkSumEquals(matrix_sum - boson_product,
                   matrix_sum_expected - boson_product_expected, 3);
    checkSumEquals(boson_sum - matrix_product,
                   boson_sum_expected - matrix_product_expected, 3);
    checkSumEquals(spin_sum - boson_product,
                   spin_sum_expected - boson_product_expected, 3);
    checkSumEquals(boson_sum - spin_product,
                   boson_sum_expected - spin_product_expected, 3);
  }

  // `product - sum`
  {
    checkSumEquals(matrix_product - matrix_sum,
                   matrix_product_expected - matrix_sum_expected, 3);
    checkSumEquals(spin_product - spin_sum,
                   spin_product_expected - spin_sum_expected, 3);
    checkSumEquals(boson_product - boson_sum,
                   boson_product_expected - boson_sum_expected, 3);
    checkSumEquals(matrix_product - spin_sum,
                   matrix_product_expected - spin_sum_expected, 3);
    checkSumEquals(spin_product - matrix_sum,
                   spin_product_expected - matrix_sum_expected, 3);
    checkSumEquals(matrix_product - boson_sum,
                   matrix_product_expected - boson_sum_expected, 3);
    checkSumEquals(boson_product - matrix_sum,
                   boson_product_expected - matrix_sum_expected, 3);
    checkSumEquals(spin_product - boson_sum,
                   spin_product_expected - boson_sum_expected, 3);
    checkSumEquals(boson_product - spin_sum,
                   boson_product_expected - spin_sum_expected, 3);
  }

  // `sum - sum`
  {
    checkSumEquals(matrix_sum - matrix_sum,
                   matrix_sum_expected - matrix_sum_expected, 2);
    checkSumEquals(spin_sum - spin_sum, spin_sum_expected - spin_sum_expected,
                   2);
    checkSumEquals(boson_sum - boson_sum,
                   boson_sum_expected - boson_sum_expected, 2);
    checkSumEquals(matrix_sum - spin_sum,
                   matrix_sum_expected - spin_sum_expected);
    checkSumEquals(spin_sum - matrix_sum,
                   spin_sum_expected - matrix_sum_expected);
    checkSumEquals(matrix_sum - boson_sum,
                   matrix_sum_expected - boson_sum_expected);
    checkSumEquals(boson_sum - matrix_sum,
                   boson_sum_expected - matrix_sum_expected);
    checkSumEquals(spin_sum - boson_sum,
                   spin_sum_expected - boson_sum_expected);
    checkSumEquals(boson_sum - spin_sum,
                   boson_sum_expected - spin_sum_expected);
  }

  // `sum * product`
  {
    checkSumEquals(matrix_sum * matrix_product,
                   matrix_sum_expected * matrix_product_expected, 2);
    checkSumEquals(spin_sum * spin_product,
                   spin_sum_expected * spin_product_expected, 2);
    checkSumEquals(boson_sum * boson_product,
                   boson_sum_expected * boson_product_expected, 2);
    checkSumEquals(matrix_sum * spin_product,
                   matrix_sum_expected * spin_product_expected, 2);
    checkSumEquals(spin_sum * matrix_product,
                   spin_sum_expected * matrix_product_expected, 2);
    checkSumEquals(matrix_sum * boson_product,
                   matrix_sum_expected * boson_product_expected, 2);
    checkSumEquals(boson_sum * matrix_product,
                   boson_sum_expected * matrix_product_expected, 2);
    checkSumEquals(spin_sum * boson_product,
                   spin_sum_expected * boson_product_expected, 2);
    checkSumEquals(boson_sum * spin_product,
                   boson_sum_expected * spin_product_expected, 2);
  }

  // `product * sum`
  {
    checkSumEquals(matrix_product * matrix_sum,
                   matrix_product_expected * matrix_sum_expected, 2);
    checkSumEquals(spin_product * spin_sum,
                   spin_product_expected * spin_sum_expected, 2);
    checkSumEquals(boson_product * boson_sum,
                   boson_product_expected * boson_sum_expected, 2);
    checkSumEquals(matrix_product * spin_sum,
                   matrix_product_expected * spin_sum_expected, 2);
    checkSumEquals(spin_product * matrix_sum,
                   spin_product_expected * matrix_sum_expected, 2);
    checkSumEquals(matrix_product * boson_sum,
                   matrix_product_expected * boson_sum_expected, 2);
    checkSumEquals(boson_product * matrix_sum,
                   boson_product_expected * matrix_sum_expected, 2);
    checkSumEquals(spin_product * boson_sum,
                   spin_product_expected * boson_sum_expected, 2);
    checkSumEquals(boson_product * spin_sum,
                   boson_product_expected * spin_sum_expected, 2);
  }

  // `sum * sum`
  {
    checkSumEquals(matrix_sum * matrix_sum,
                   matrix_sum_expected * matrix_sum_expected, 3);
    checkSumEquals(spin_sum * spin_sum, spin_sum_expected * spin_sum_expected,
                   3);
    checkSumEquals(boson_sum * boson_sum,
                   boson_sum_expected * boson_sum_expected, 3);
    checkSumEquals(matrix_sum * spin_sum,
                   matrix_sum_expected * spin_sum_expected);
    checkSumEquals(spin_sum * matrix_sum,
                   spin_sum_expected * matrix_sum_expected);
    checkSumEquals(matrix_sum * boson_sum,
                   matrix_sum_expected * boson_sum_expected);
    checkSumEquals(boson_sum * matrix_sum,
                   boson_sum_expected * matrix_sum_expected);
    checkSumEquals(spin_sum * boson_sum,
                   spin_sum_expected * boson_sum_expected);
    checkSumEquals(boson_sum * spin_sum,
                   boson_sum_expected * spin_sum_expected);
  }

  // `sum += product`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 += matrix_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + matrix_product_expected,
                   3);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 += spin_product;
    checkSumEquals(spin_sum_0, spin_sum_expected + spin_product_expected, 3);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 += boson_product;
    checkSumEquals(boson_sum_0, boson_sum_expected + boson_product_expected, 3);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 += spin_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + spin_product_expected,
                   3);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 += boson_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + boson_product_expected,
                   3);
  }

  // `sum += sum`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 += matrix_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + matrix_sum_expected, 2);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 += spin_sum;
    checkSumEquals(spin_sum_0, spin_sum_expected + spin_sum_expected, 2);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 += boson_sum;
    checkSumEquals(boson_sum_0, boson_sum_expected + boson_sum_expected, 2);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 += spin_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + spin_sum_expected);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 += boson_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected + boson_sum_expected);
  }

  // `sum -= product`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= matrix_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - matrix_product_expected,
                   3);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 -= spin_product;
    checkSumEquals(spin_sum_0, spin_sum_expected - spin_product_expected, 3);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 -= boson_product;
    checkSumEquals(boson_sum_0, boson_sum_expected - boson_product_expected, 3);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= spin_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - spin_product_expected,
                   3);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= boson_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - boson_product_expected,
                   3);
  }

  // `sum -= sum`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= matrix_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - matrix_sum_expected, 2);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 -= spin_sum;
    checkSumEquals(spin_sum_0, spin_sum_expected - spin_sum_expected, 2);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 -= boson_sum;
    checkSumEquals(boson_sum_0, boson_sum_expected - boson_sum_expected, 2);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= spin_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - spin_sum_expected);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 -= boson_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected - boson_sum_expected);
  }

  // `sum *= product`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= matrix_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * matrix_product_expected,
                   2);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 *= spin_product;
    checkSumEquals(spin_sum_0, spin_sum_expected * spin_product_expected, 2);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 *= boson_product;
    checkSumEquals(boson_sum_0, boson_sum_expected * boson_product_expected, 2);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= spin_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * spin_product_expected,
                   2);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= boson_product;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * boson_product_expected,
                   2);
  }

  // `sum *= sum`
  {
    auto matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= matrix_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * matrix_sum_expected, 3);

    auto spin_sum_0 = spin_sum;
    spin_sum_0 *= spin_sum;
    checkSumEquals(spin_sum_0, spin_sum_expected * spin_sum_expected, 3);

    auto boson_sum_0 = boson_sum;
    boson_sum_0 *= boson_sum;
    checkSumEquals(boson_sum_0, boson_sum_expected * boson_sum_expected, 3);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= spin_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * spin_sum_expected);

    matrix_sum_0 = matrix_sum;
    matrix_sum_0 *= boson_sum;
    checkSumEquals(matrix_sum_0, matrix_sum_expected * boson_sum_expected);
  }
}
