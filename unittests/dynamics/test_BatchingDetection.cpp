// /*******************************************************************************
//  * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "BatchingUtils.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

TEST(BatchingDetectionTester, checkPositive) {
  {
    // Simple case with different coefficients
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.5 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));
    EXPECT_TRUE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
  {
    // swapping the order of the terms should not matter
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 =
        (2 * M_PI * 0.5 *
         (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
          cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1))) +
        (2 * M_PI * cudaq::matrix_op::number(1)) +
        (2 * M_PI * cudaq::matrix_op::number(0));
    EXPECT_TRUE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
  {
    // Different operators (on same degree of freedom) should not matter
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.5 *
                 (cudaq::spin::x(0) * cudaq::boson_op::create(1) +
                  cudaq::spin::z(0) * cudaq::boson_op::annihilate(1)));
    EXPECT_TRUE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }

  {
    // callback coefficient
    auto mod_func =
        [](const cudaq::parameter_map &params) -> std::complex<double> {
      auto it = params.find("t");
      if (it != params.end()) {
        double t = it->second.real();
        const auto result = std::cos(2 * M_PI * t);
        return result;
      }
      throw std::runtime_error("Cannot find the time parameter.");
    };
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (mod_func *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));
    EXPECT_TRUE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }

  {
    // user-defined op
    auto displacement_matrix =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>>
               &parameters) -> cudaq::complex_matrix {
      std::size_t dimension = dimensions[0];
      auto entry = parameters.find("displacement");
      if (entry == parameters.end())
        throw std::runtime_error("missing value for parameter 'displacement'");
      auto displacement_amplitude = entry->second;
      auto create = cudaq::complex_matrix(dimension, dimension);
      auto annihilate = cudaq::complex_matrix(dimension, dimension);
      for (std::size_t i = 0; i + 1 < dimension; i++) {
        create[{i + 1, i}] = std::sqrt(static_cast<double>(i + 1));
        annihilate[{i, i + 1}] = std::sqrt(static_cast<double>(i + 1));
      }
      auto term1 = displacement_amplitude * create;
      auto term2 = std::conj(displacement_amplitude) * annihilate;
      return (term1 - term2).exponential();
    };

    cudaq::matrix_handler::define("my_displace_op", {-1}, displacement_matrix);

    // Instantiate a displacement operator acting on the given degree of
    // freedom.
    auto displacement = [](std::size_t degree) {
      return cudaq::matrix_handler::instantiate("my_displace_op", {degree});
    };
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.5 *
                 (cudaq::boson_op::annihilate(0) * displacement(1) +
                  displacement(0) * cudaq::boson_op::annihilate(1)));
    EXPECT_TRUE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
}

TEST(BatchingDetectionTester, checkNegative) {
  {
    // Different number of product terms
    cudaq::sum_op<cudaq::matrix_handler> ham1(2 * M_PI * cudaq::spin_op::x(0));
    cudaq::sum_op<cudaq::matrix_handler> ham2(2 * M_PI * cudaq::spin_op::x(0) +
                                              2 * M_PI * cudaq::spin_op::y(0));
    EXPECT_FALSE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
  {
    // Term acts on different degrees of freedom
    // Simple case with different coefficients
    auto ham1 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.25 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(1) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(1)));

    auto ham2 = (2 * M_PI * cudaq::matrix_op::number(1)) +
                (2 * M_PI * cudaq::matrix_op::number(0)) +
                (2 * M_PI * 0.5 *
                 (cudaq::boson_op::annihilate(0) * cudaq::boson_op::create(2) +
                  cudaq::boson_op::create(0) * cudaq::boson_op::annihilate(2)));
    EXPECT_FALSE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
  {
    // Same degrees, but not compatible (2-body op vs multiplication of 1-body
    // ops)
    auto two_body_matrix =
        [](const std::vector<int64_t> &dimensions,
           const std::unordered_map<std::string, std::complex<double>>
               &parameters) -> cudaq::complex_matrix {
      if (dimensions.size() != 2)
        throw std::runtime_error("Expected two dimensions for two-body op");
      std::size_t dimension = dimensions[0] * dimensions[1];
      return cudaq::complex_matrix(dimension, dimension);
    };

    cudaq::matrix_handler::define("my_two_body_op", {-1}, two_body_matrix);

    auto two_body_op = [](const std::vector<std::size_t> degrees) {
      return cudaq::matrix_handler::instantiate("my_two_body_op", degrees);
    };

    cudaq::sum_op<cudaq::matrix_handler> ham1(cudaq::spin::x(0) *
                                              cudaq::spin::z(1));
    cudaq::sum_op<cudaq::matrix_handler> ham2(two_body_op({0, 1}));
    EXPECT_FALSE(cudaq::__internal__::checkBatchingCompatibility({ham1, ham2}));
  }
}
