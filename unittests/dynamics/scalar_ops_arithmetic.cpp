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

TEST(OperatorExpressions, checkScalarOpsArithmeticComplex) {
  // Arithmetic overloads against complex doubles.
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

  // + : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto new_scalar_op = value_1 + scalar_op;
    // function + scalar_op;
    auto reverse_order_op = scalar_op + value_1;
    EXPECT_NEAR(std::abs(scalar_op.evaluate({})), std::abs(value_0), 1e-5);

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});
    auto want_value = value_1 + value_0;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op + reverse_order_op;
    auto got_value_third = third_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value + want_value),
                1e-5);
  }

  // + : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);

    auto new_scalar_op = value_0 + scalar_op;
    auto reverse_order_op = scalar_op + value_0;

    auto got_value = new_scalar_op.evaluate({{"value", value_1}});
    auto got_value_1 = reverse_order_op.evaluate({{"value", value_1}});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 + value_0), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op + reverse_order_op;
    auto got_value_third = third_op.evaluate({{"value", value_1}});
    auto want_value = value_0 + value_1 + value_1 + value_0;
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // - : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_1);

    auto new_scalar_op = value_3 - scalar_op;
    auto reverse_order_op = scalar_op - value_3;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 - value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 - value_3), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op - reverse_order_op;
    auto got_value_third = third_op.evaluate({});
    auto want_value = (value_3 - value_1) - (value_1 - value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // - : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);

    auto new_scalar_op = value_2 - scalar_op;
    auto reverse_order_op = scalar_op - value_2;

    auto got_value = new_scalar_op.evaluate({{"value", value_1}});
    auto got_value_1 = reverse_order_op.evaluate({{"value", value_1}});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 - value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 - value_2), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op - reverse_order_op;
    auto got_value_third = third_op.evaluate({{"value", value_1}});
    auto want_value = (value_2 - value_1) - (value_1 - value_2);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // * : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);

    auto new_scalar_op = value_3 * scalar_op;
    auto reverse_order_op = scalar_op * value_3;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 * value_2), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_2 * value_3), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op * reverse_order_op;
    auto got_value_third = third_op.evaluate({});
    auto want_value = (value_3 * value_2) * (value_2 * value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // * : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);

    auto new_scalar_op = value_3 * scalar_op;
    auto reverse_order_op = scalar_op * value_3;

    auto got_value = new_scalar_op.evaluate({{"value", value_2}});
    auto got_value_1 = reverse_order_op.evaluate({{"value", value_2}});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 * value_2), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_2 * value_3), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op * reverse_order_op;
    auto got_value_third = third_op.evaluate({{"value", value_2}});
    auto want_value = (value_3 * value_2) * (value_2 * value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // / : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);

    auto new_scalar_op = value_3 / scalar_op;
    auto reverse_order_op = scalar_op / value_3;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 / value_2), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_2 / value_3), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op / reverse_order_op;
    auto got_value_third = third_op.evaluate({});
    auto want_value = (value_3 / value_2) / (value_2 / value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // / : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);

    auto new_scalar_op = value_3 / scalar_op;
    auto reverse_order_op = scalar_op / value_3;

    auto got_value = new_scalar_op.evaluate({{"value", value_1}});
    auto got_value_1 = reverse_order_op.evaluate({{"value", value_1}});

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 / value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 / value_3), 1e-5);

    // Checking composition of many scalar operators.
    auto third_op = new_scalar_op / reverse_order_op;
    auto got_value_third = third_op.evaluate({{"value", value_1}});
    auto want_value = (value_3 / value_1) / (value_1 / value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // += : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    scalar_op += value_0;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_0), 1e-5);
  }

  // += : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op += value_1;

    auto got_value = scalar_op.evaluate({{"value", value_0}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_1), 1e-5);
  }

  // -= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    scalar_op -= value_0;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 - value_0), 1e-5);
  }

  // -= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op -= value_1;

    auto got_value = scalar_op.evaluate({{"value", value_0}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 - value_1), 1e-5);
  }

  // *= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    scalar_op *= value_3;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 * value_3), 1e-5);
  }

  // *= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op *= value_3;

    auto got_value = scalar_op.evaluate({{"value", value_2}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 * value_3), 1e-5);
  }

  // /= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    scalar_op /= value_3;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 / value_3), 1e-5);
  }

  // /= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op /= value_3;

    auto got_value = scalar_op.evaluate({{"value", value_2}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 / value_3), 1e-5);
  }
}

TEST(OperatorExpressions, checkScalarOpsArithmeticScalarOps) {
  // Arithmetic overloads against other scalar ops.
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

  // I use another function here to make sure that local variables
  // that may be unique to each ScalarOp's generators are both kept
  // track of when we merge the generators.
  auto alternative_local_variable = true;
  auto alternative_function =
      [&](std::map<std::string, std::complex<double>> parameters) {
        if (!alternative_local_variable)
          throw std::runtime_error("Local variable not detected.");
        return parameters["other"];
      };

  // + : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto other_scalar_op = cudaq::scalar_operator(value_1);

    auto new_scalar_op = other_scalar_op + scalar_op;
    auto reverse_order_op = scalar_op + other_scalar_op;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});
    auto want_value = value_1 + value_0;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value), 1e-5);
  }

  // + : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    auto other_scalar_op = cudaq::scalar_operator(alternative_function);

    auto new_scalar_op = other_scalar_op + scalar_op;
    auto reverse_order_op = scalar_op + other_scalar_op;

    std::map<std::string, std::complex<double>> parameter_map = {
        {"value", value_1}, {"other", value_0}};

    auto got_value = new_scalar_op.evaluate(parameter_map);
    auto got_value_1 = reverse_order_op.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 + value_0), 1e-5);
  }

  // - : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    auto other_scalar_op = cudaq::scalar_operator(value_1);

    auto new_scalar_op = other_scalar_op - scalar_op;
    auto reverse_order_op = scalar_op - other_scalar_op;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});
    auto want_value = value_1 - value_2;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value), 1e-5);
  }

  // - : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    auto other_scalar_op = cudaq::scalar_operator(alternative_function);

    auto new_scalar_op = other_scalar_op - scalar_op;
    auto reverse_order_op = scalar_op - other_scalar_op;

    std::map<std::string, std::complex<double>> parameter_map = {
        {"value", value_1}, {"other", value_3}};

    auto got_value = new_scalar_op.evaluate(parameter_map);
    auto got_value_1 = reverse_order_op.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 - value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 - value_3), 1e-5);
  }

  // * : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    auto other_scalar_op = cudaq::scalar_operator(value_3);

    auto new_scalar_op = other_scalar_op * scalar_op;
    auto reverse_order_op = scalar_op * other_scalar_op;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});
    auto want_value = value_3 * value_2;
    auto reverse_want_value = value_2 * value_3;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(reverse_want_value), 1e-5);
  }

  // * : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    auto other_scalar_op = cudaq::scalar_operator(alternative_function);

    auto new_scalar_op = other_scalar_op * scalar_op;
    auto reverse_order_op = scalar_op * other_scalar_op;

    std::map<std::string, std::complex<double>> parameter_map = {
        {"value", value_1}, {"other", value_3}};

    auto got_value = new_scalar_op.evaluate(parameter_map);
    auto got_value_1 = reverse_order_op.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 * value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 * value_3), 1e-5);
  }

  // / : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto other_scalar_op = cudaq::scalar_operator(value_2);

    auto new_scalar_op = other_scalar_op / scalar_op;
    auto reverse_order_op = scalar_op / other_scalar_op;

    auto got_value = new_scalar_op.evaluate({});
    auto got_value_1 = reverse_order_op.evaluate({});
    auto want_value = value_2 / value_0;
    auto reverse_want_value = value_0 / value_2;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(reverse_want_value), 1e-5);
  }

  // / : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    auto other_scalar_op = cudaq::scalar_operator(alternative_function);

    auto new_scalar_op = other_scalar_op / scalar_op;
    auto reverse_order_op = scalar_op / other_scalar_op;

    std::map<std::string, std::complex<double>> parameter_map = {
        {"value", value_0}, {"other", value_3}};

    auto got_value = new_scalar_op.evaluate(parameter_map);
    auto got_value_1 = reverse_order_op.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 / value_0), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_0 / value_3), 1e-5);
  }

  // += : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto other = cudaq::scalar_operator(value_0);
    scalar_op += other;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_0), 1e-5);
  }

  // += : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    auto other = cudaq::scalar_operator(value_1);
    scalar_op += other;

    auto scalar_op_1 = cudaq::scalar_operator(function);
    auto other_function = cudaq::scalar_operator(alternative_function);
    scalar_op_1 += other_function;

    auto got_value = scalar_op.evaluate({{"value", value_0}});
    auto got_value_1 =
        scalar_op_1.evaluate({{"value", value_0}, {"other", value_1}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 + value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_0 + value_1), 1e-5);
  }

  // -= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    scalar_op -= value_0;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 - value_0), 1e-5);
  }

  // -= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op -= value_1;

    auto got_value = scalar_op.evaluate({{"value", value_0}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_0 - value_1), 1e-5);
  }

  // *= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    scalar_op *= value_3;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 * value_3), 1e-5);
  }

  // *= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op *= value_3;

    auto got_value = scalar_op.evaluate({{"value", value_2}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 * value_3), 1e-5);
  }

  // /= : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_2);
    scalar_op /= value_3;

    auto got_value = scalar_op.evaluate({});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 / value_3), 1e-5);
  }

  // /= : Scalar operator from lambda.
  {
    auto scalar_op = cudaq::scalar_operator(function);
    scalar_op /= value_3;

    auto got_value = scalar_op.evaluate({{"value", value_2}});
    EXPECT_NEAR(std::abs(got_value), std::abs(value_2 / value_3), 1e-5);
  }
}