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

std::complex<double> get_value(cudaq::parameter_map params) {
  return params["value"];
}

cudaq::scalar_operator negate(cudaq::scalar_operator op) { return -1.0 * op; }

TEST(OperatorExpressions, checkScalarOpsUnary) {
  auto scalar = cudaq::scalar_operator(1.0);
  EXPECT_EQ((+scalar).evaluate(), std::complex<double>(1.0));
  EXPECT_EQ((-scalar).evaluate(), std::complex<double>(-1.0));
  EXPECT_EQ(negate(scalar).evaluate(), std::complex<double>(-1.0));
  EXPECT_EQ(scalar.evaluate(), std::complex<double>(1.0));
}

TEST(OperatorExpressions, checkScalarOpsSimpleComplex) {

  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  // From concrete values.
  {
    auto operator_0 = cudaq::scalar_operator(value_0);
    auto operator_1 = cudaq::scalar_operator(value_1);
    auto operator_2 = cudaq::scalar_operator(value_2);
    auto operator_3 = cudaq::scalar_operator(value_3);

    auto got_value_0 = operator_0.evaluate();
    auto got_value_1 = operator_1.evaluate();
    auto got_value_2 = operator_2.evaluate();
    auto got_value_3 = operator_3.evaluate();

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }

  // From a lambda function.
  {
    auto function =
        [](const std::unordered_map<std::string, std::complex<double>>
               &parameters) {
          auto entry = parameters.find("value");
          if (entry == parameters.end())
            throw std::runtime_error("value not defined in parameters");
          return entry->second;
        };

    std::unordered_map<std::string, std::complex<double>> parameter_map;

    auto operator_0 = cudaq::scalar_operator(function);
    auto operator_1 = cudaq::scalar_operator(function);
    auto operator_2 = cudaq::scalar_operator(function);
    auto operator_3 = cudaq::scalar_operator(function);

    parameter_map["value"] = value_0;
    auto got_value_0 = operator_0.evaluate(parameter_map);
    parameter_map["value"] = value_1;
    auto got_value_1 = operator_1.evaluate(parameter_map);
    parameter_map["value"] = value_2;
    auto got_value_2 = operator_2.evaluate(parameter_map);
    parameter_map["value"] = value_3;
    auto got_value_3 = operator_3.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }
}

TEST(OperatorExpressions, checkScalarOpsSimpleDouble) {

  double value_0 = 0.1;
  double value_1 = 0.2;
  double value_2 = 2.1;
  double value_3 = 2.2;

  // From concrete values.
  {
    auto operator_0 = cudaq::scalar_operator(value_0);
    auto operator_1 = cudaq::scalar_operator(value_1);
    auto operator_2 = cudaq::scalar_operator(value_2);
    auto operator_3 = cudaq::scalar_operator(value_3);

    auto got_value_0 = operator_0.evaluate();
    auto got_value_1 = operator_1.evaluate();
    auto got_value_2 = operator_2.evaluate();
    auto got_value_3 = operator_3.evaluate();

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }

  // From a lambda function.
  {
    auto function =
        [](const std::unordered_map<std::string, std::complex<double>>
               &parameters) {
          auto entry = parameters.find("value");
          if (entry == parameters.end())
            throw std::runtime_error("value not defined in parameters");
          return entry->second;
        };

    std::unordered_map<std::string, std::complex<double>> parameter_map;

    auto operator_0 = cudaq::scalar_operator(function);
    auto operator_1 = cudaq::scalar_operator(function);
    auto operator_2 = cudaq::scalar_operator(function);
    auto operator_3 = cudaq::scalar_operator(function);

    parameter_map["value"] = value_0;
    auto got_value_0 = operator_0.evaluate(parameter_map);
    parameter_map["value"] = value_1;
    auto got_value_1 = operator_1.evaluate(parameter_map);
    parameter_map["value"] = value_2;
    auto got_value_2 = operator_2.evaluate(parameter_map);
    parameter_map["value"] = value_3;
    auto got_value_3 = operator_3.evaluate(parameter_map);

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }
}

TEST(OperatorExpressions, checkScalarOpsArithmeticComplex) {
  // Arithmetic overloads against complex doubles.
  std::complex<double> value_0 = 0.1 + 0.1;
  std::complex<double> value_1 = 0.1 + 1.0;
  std::complex<double> value_2 = 2.0 + 0.1;
  std::complex<double> value_3 = 2.0 + 1.0;

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  // + : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);

    auto new_scalar_op = value_1 + scalar_op;
    auto reverse_order_op = scalar_op + value_1;
    EXPECT_NEAR(std::abs(scalar_op.evaluate()), std::abs(value_0), 1e-5);

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();
    auto want_value = value_1 + value_0;

    EXPECT_NEAR(std::abs(got_value), std::abs(want_value), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(want_value), 1e-5);

    auto third_op = new_scalar_op + reverse_order_op;
    auto got_value_third = third_op.evaluate();
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 - value_1), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_1 - value_3), 1e-5);

    auto third_op = new_scalar_op - reverse_order_op;
    auto got_value_third = third_op.evaluate();
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 * value_2), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_2 * value_3), 1e-5);

    auto third_op = new_scalar_op * reverse_order_op;
    auto got_value_third = third_op.evaluate();
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();

    EXPECT_NEAR(std::abs(got_value), std::abs(value_3 / value_2), 1e-5);
    EXPECT_NEAR(std::abs(got_value_1), std::abs(value_2 / value_3), 1e-5);

    auto third_op = new_scalar_op / reverse_order_op;
    auto got_value_third = third_op.evaluate();
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

    auto third_op = new_scalar_op / reverse_order_op;
    auto got_value_third = third_op.evaluate({{"value", value_1}});
    auto want_value = (value_3 / value_1) / (value_1 / value_3);
    EXPECT_NEAR(std::abs(got_value_third), std::abs(want_value), 1e-5);
  }

  // += : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    scalar_op += value_0;

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

  auto function = [](const std::unordered_map<std::string, std::complex<double>>
                         &parameters) {
    auto entry = parameters.find("value");
    if (entry == parameters.end())
      throw std::runtime_error("value not defined in parameters");
    return entry->second;
  };

  // I use another function here to make sure that local variables
  // that may be unique to each ScalarOp's generators are both kept
  // track of when we merge the generators.
  auto alternative_function =
      [](const std::unordered_map<std::string, std::complex<double>>
             &parameters) {
        auto entry = parameters.find("other");
        if (entry == parameters.end())
          throw std::runtime_error("other not defined in parameters");
        return entry->second;
      };

  // + : Constant scalar operator.
  {
    auto scalar_op = cudaq::scalar_operator(value_0);
    auto other_scalar_op = cudaq::scalar_operator(value_1);

    auto new_scalar_op = other_scalar_op + scalar_op;
    auto reverse_order_op = scalar_op + other_scalar_op;

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();
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

    std::unordered_map<std::string, std::complex<double>> parameter_map = {
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();
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

    std::unordered_map<std::string, std::complex<double>> parameter_map = {
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();
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

    std::unordered_map<std::string, std::complex<double>> parameter_map = {
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

    auto got_value = new_scalar_op.evaluate();
    auto got_value_1 = reverse_order_op.evaluate();
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

    std::unordered_map<std::string, std::complex<double>> parameter_map = {
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

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

    auto got_value = scalar_op.evaluate();
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

TEST(OperatorExpressions, checkScalarOpsFromFunctions) {

  std::complex<double> value = 2.;
  std::complex<double> squeeze_ampl;
  cudaq::parameter_map params = {{"squeezing", squeeze_ampl}, {"value", value}};
  cudaq::dimension_map dims = {{1, 3}};
  auto squeeze = cudaq::matrix_op::squeeze(1);
  auto squeeze_mat = utils::squeeze_matrix(3, squeeze_ampl);
  auto id_mat = utils::id_matrix(3);
  auto sum = squeeze + cudaq::matrix_op::number(1);
  auto sum_mat = squeeze_mat + utils::number_matrix(3);

  // matrix operator + lambda
  {
    auto prod_res =
        squeeze + [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } +
                        squeeze;
    auto prod_want = value * id_mat + squeeze_mat;

    auto sum_res = sum + [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } +
                       sum;
    auto sum_want = value * id_mat + sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want);
  }

  // matrix operator - lambda
  {
    auto prod_res =
        squeeze - [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } -
                        squeeze;
    auto prod_want = squeeze_mat - value * id_mat;
    auto prod_want_rev = value * id_mat - squeeze_mat;

    auto sum_res = sum - [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } -
                       sum;
    auto sum_want = sum_mat - value * id_mat;
    auto sum_want_rev = value * id_mat - sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want_rev);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want_rev);
  }

  // matrix operator * lambda
  {
    auto prod_res =
        squeeze * [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } *
                        squeeze;
    auto prod_want = value * squeeze_mat;

    auto sum_res = sum * [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_res_rev = [](cudaq::parameter_map ps) { return ps["value"]; } *
                       sum;
    auto sum_want = value * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want);
  }

  // matrix operator / lambda
  {
    auto prod_res =
        squeeze / [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_want = (1. / value) * squeeze_mat;

    auto sum_res = sum / [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_want = (1. / value) * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator += lambda
  {
    auto sum_res = sum;
    sum_res += [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_want = value * id_mat + sum_mat;

    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator -= lambda
  {
    auto sum_res = sum;
    sum_res -= [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_want = sum_mat - value * id_mat;

    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator *= lambda
  {
    auto prod_res = squeeze;
    prod_res *= [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_want = value * squeeze_mat;

    auto sum_res = sum;
    sum_res *= [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_want = value * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator /= lambda
  {
    auto prod_res = squeeze;
    prod_res /= [](cudaq::parameter_map ps) { return ps["value"]; };
    auto prod_want = (1. / value) * squeeze_mat;

    auto sum_res = sum;
    sum_res /= [](cudaq::parameter_map ps) { return ps["value"]; };
    auto sum_want = (1. / value) * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator + function
  {
    auto prod_res = squeeze + get_value;
    auto prod_res_rev = get_value + squeeze;
    auto prod_want = value * id_mat + squeeze_mat;

    auto sum_res = sum + get_value;
    auto sum_res_rev = get_value + sum;
    auto sum_want = value * id_mat + sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want);
  }

  // matrix operator - function
  {
    auto prod_res = squeeze - get_value;
    auto prod_res_rev = get_value - squeeze;
    auto prod_want = squeeze_mat - value * id_mat;
    auto prod_want_rev = value * id_mat - squeeze_mat;

    auto sum_res = sum - get_value;
    auto sum_res_rev = get_value - sum;
    auto sum_want = sum_mat - value * id_mat;
    auto sum_want_rev = value * id_mat - sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want_rev);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want_rev);
  }

  // matrix operator * function
  {
    auto prod_res = squeeze * get_value;
    auto prod_res_rev = get_value * squeeze;
    auto prod_want = value * squeeze_mat;

    auto sum_res = sum * get_value;
    auto sum_res_rev = get_value * sum;
    auto sum_want = value * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(prod_res_rev.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
    utils::checkEqual(sum_res_rev.to_matrix(dims, params), sum_want);
  }

  // matrix operator / function
  {
    auto prod_res = squeeze / get_value;
    auto prod_want = (1. / value) * squeeze_mat;

    auto sum_res = sum / get_value;
    auto sum_want = (1. / value) * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator += function
  {
    auto sum_res = sum;
    sum_res += get_value;
    auto sum_want = value * id_mat + sum_mat;

    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator -= function
  {
    auto sum_res = sum;
    sum_res -= get_value;
    auto sum_want = sum_mat - value * id_mat;

    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator *= function
  {
    auto prod_res = squeeze;
    prod_res *= get_value;
    auto prod_want = value * squeeze_mat;

    auto sum_res = sum;
    sum_res *= get_value;
    auto sum_want = value * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }

  // matrix operator /= function
  {
    auto prod_res = squeeze;
    prod_res /= get_value;
    auto prod_want = (1. / value) * squeeze_mat;

    auto sum_res = sum;
    sum_res /= get_value;
    auto sum_want = (1. / value) * sum_mat;

    utils::checkEqual(prod_res.to_matrix(dims, params), prod_want);
    utils::checkEqual(sum_res.to_matrix(dims, params), sum_want);
  }
}
