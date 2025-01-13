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

    auto got_value_0 = operator_0.evaluate({});
    auto got_value_1 = operator_1.evaluate({});
    auto got_value_2 = operator_2.evaluate({});
    auto got_value_3 = operator_3.evaluate({});

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }

  // From a lambda function.
  {
    auto function = [](std::map<std::string, std::complex<double>> parameters) {
      return parameters["value"];
    };

    std::map<std::string, std::complex<double>> parameter_map;

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

    auto got_value_0 = operator_0.evaluate({});
    auto got_value_1 = operator_1.evaluate({});
    auto got_value_2 = operator_2.evaluate({});
    auto got_value_3 = operator_3.evaluate({});

    EXPECT_NEAR(std::abs(value_0), std::abs(got_value_0), 1e-5);
    EXPECT_NEAR(std::abs(value_1), std::abs(got_value_1), 1e-5);
    EXPECT_NEAR(std::abs(value_2), std::abs(got_value_2), 1e-5);
    EXPECT_NEAR(std::abs(value_3), std::abs(got_value_3), 1e-5);
  }

  // From a lambda function.
  {
    auto function = [](std::map<std::string, std::complex<double>> parameters) {
      return parameters["value"];
    };

    std::map<std::string, std::complex<double>> parameter_map;

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