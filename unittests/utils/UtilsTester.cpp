/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "common/FmtCore.h"
#include "common/JsonConvert.h"
#include "cudaq/utils/cudaq_utils.h"

TEST(UtilsTester, checkRange) {
  {
    auto v = cudaq::range(0, 10, 2);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(5, v.size());
    std::vector<int> expected{0, 2, 4, 6, 8};
    EXPECT_EQ(expected, v);
  }

  {
    auto v = cudaq::range(10);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(10, v.size());
    std::vector<int> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(expected, v);
  }

  {
    auto v = cudaq::range(10, 2, -1);
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(8, v.size());
    std::vector<int> expected{10, 9, 8, 7, 6, 5, 4, 3};
    EXPECT_EQ(expected, v);
  }

  {
    std::vector<std::size_t> nothing(10);
    // Common pattern, implement size_t overload to
    // avoid user static_casts.
    EXPECT_ANY_THROW({ auto v = cudaq::range((std::size_t)-1); });
    auto v = cudaq::range(nothing.size());
    std::cout << fmt::format("{}", fmt::join(v, ",")) << "\n";
    EXPECT_EQ(10, v.size());
    std::vector<std::size_t> expected{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    EXPECT_EQ(expected, v);
  }
}

TEST(UtilsTester, JsonSerDesOptimizer) {
  {
    cudaq::optimizers::cobyla test_opt;
    test_opt.f_tol = 0.01;
    test_opt.lower_bounds = {0.02, 0.03};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"f_tol\":0.01,\"lower_bounds\":[0.02,0.03]}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(), "\"COBYLA\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::neldermead test_opt;
    test_opt.f_tol = 0.01;
    test_opt.upper_bounds = {0.02, 0.03};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"f_tol\":0.01,\"upper_bounds\":[0.02,0.03]}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(),
              "\"NELDERMEAD\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::lbfgs test_opt;
    test_opt.step_size = 0.99;
    test_opt.initial_parameters = {0.04, -0.05};
    test_opt.max_line_search_trials = 12;
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"initial_parameters\":[0.04,-0.05],\"max_line_"
                        "search_trials\":12,\"step_size\":0.99}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(), "\"LBFGS\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::spsa test_opt;
    test_opt.step_size = 0.99;
    test_opt.initial_parameters = {0.04, -0.05};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(),
              "{\"initial_parameters\":[0.04,-0.05],\"step_size\":0.99}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(), "\"SPSA\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::adam test_opt;
    test_opt.step_size = 0.99;
    test_opt.initial_parameters = {0.04, -0.05};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(),
              "{\"initial_parameters\":[0.04,-0.05],\"step_size\":0.99}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(), "\"ADAM\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::gradient_descent test_opt;
    test_opt.step_size = 0.99;
    test_opt.initial_parameters = {0.04, -0.05};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(),
              "{\"initial_parameters\":[0.04,-0.05],\"step_size\":0.99}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(),
              "\"GRAD_DESC\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::optimizers::sgd test_opt;
    test_opt.step_size = 0.99;
    test_opt.initial_parameters = {0.04, -0.05};
    json j(test_opt);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(),
              "{\"initial_parameters\":[0.04,-0.05],\"step_size\":0.99}");
    EXPECT_EQ(json(cudaq::get_optimizer_type(test_opt)).dump(), "\"SGD\"");

    auto test_opt_round_trip =
        make_optimizer_from_json(j, cudaq::get_optimizer_type(test_opt));
    json j2(*test_opt_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }
}

TEST(UtilsTester, JsonSerDesGradient) {
  {
    cudaq::gradients::central_difference grad;
    grad.step = 0.01;
    json j(grad);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"step\":0.01}");
    EXPECT_EQ(json(cudaq::get_gradient_type(grad)).dump(), "\"CENTRAL_DIFF\"");

    auto test_grad_round_trip =
        make_gradient_from_json(j, cudaq::get_gradient_type(grad));
    json j2(*test_grad_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::gradients::forward_difference grad;
    grad.step = 0.02;
    json j(grad);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"step\":0.02}");
    EXPECT_EQ(json(cudaq::get_gradient_type(grad)).dump(), "\"FORWARD_DIFF\"");

    auto test_grad_round_trip =
        make_gradient_from_json(j, cudaq::get_gradient_type(grad));
    json j2(*test_grad_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }

  {
    cudaq::gradients::parameter_shift grad;
    grad.shiftScalar = 0.03;
    json j(grad);
    std::cout << j.dump() << '\n';
    EXPECT_EQ(j.dump(), "{\"shiftScalar\":0.03}");
    EXPECT_EQ(json(cudaq::get_gradient_type(grad)).dump(),
              "\"PARAMETER_SHIFT\"");

    auto test_grad_round_trip =
        make_gradient_from_json(j, cudaq::get_gradient_type(grad));
    json j2(*test_grad_round_trip);
    EXPECT_EQ(j.dump(), j2.dump());
  }
}
