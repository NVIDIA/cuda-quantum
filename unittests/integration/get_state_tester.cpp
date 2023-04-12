/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>
#include <fmt/core.h>
#include <numeric>

#include <iostream>

using namespace cudaq;

CUDAQ_TEST(GetStateTester, checkSimple) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    cx(q, r);
  };

  auto state = cudaq::get_state(kernel);
  state.dump();
#ifdef CUDAQ_BACKEND_DM
  EXPECT_NEAR(0.5, state(0, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(0, 3).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 0).real(), 1e-3);
  EXPECT_NEAR(0.5, state(3, 3).real(), 1e-3);
#else
  EXPECT_NEAR(1. / std::sqrt(2.), state[0].real(), 1e-3);
  EXPECT_NEAR(0., state[1].real(), 1e-3);
  EXPECT_NEAR(0., state[2].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state[3].real(), 1e-3);
#endif

  EXPECT_NEAR(state.overlap(state), 1.0, 1e-3);

  // Demonstrate a useful use-case for get_state,
  // specifically, let's approximate another 2-qubit state with a
  // general so4 rotation. Here we'll see if we can find rotational
  // parameters that create a circuit producing the bell state.
  auto so4 = [](std::vector<double> parameters) __qpu__ {
    cudaq::qubit q, r;
    ry(parameters[0], q);
    ry(parameters[1], r);

    z<cudaq::ctrl>(q, r);

    ry(parameters[2], q);
    ry(parameters[3], r);

    z<cudaq::ctrl>(q, r);

    ry(parameters[4], q);
    ry(parameters[5], r);

    z<cudaq::ctrl>(q, r);
  };

  cudaq::optimizers::cobyla optimizer;
  optimizer.max_eval = 100;
  auto [opt_val, params] = optimizer.optimize(6, [&](std::vector<double> x) {
    auto testState = cudaq::get_state(so4, x);
    return 1.0 - state.overlap(testState);
  });

  EXPECT_NEAR(opt_val, 0.0, 1e-3);
}

// CUDAQ_TEST(GetStateTester, checkGetState) {

//   auto kernel = []() __qpu__ {
//     cudaq::qubit q, r;
//     h(q);
//     cx(q, r);
//   };

//   auto state_object = cudaq::get_state(kernel);

// #ifdef CUDAQ_BACKEND_DM
//   // Check that `is_density_matrix` is true.
//   assert(state_object.is_density_matrix() == true);
//   // Can we return the density matrix as an eigen matrix?
//   auto density_matrix = state_object.get_data<Eigen::MatrixXcd>();

//   // Is a runtime error thrown if we try to return the density
//   // matrix as a vector?
//   bool error = false;
//   try {
//     auto density_vector = state_object.get_data<Eigen::VectorXcd>();
//   } catch (std::runtime_error) {
//     error = true;
//   }
//   assert(error == true);
// #else
//   // Check that `is_density_matrix` is false.
//   assert(state_object.is_density_matrix() == false);
//   // Can we return the state vector as an eigen vector?
//   auto state_vector = state_object.get_data<Eigen::VectorXcd>();
//   // Can we return the state vector as an eigen matrix?
//   auto state_matrix = state_object.get_data<Eigen::MatrixXcd>();
// #endif
// }

CUDAQ_TEST(GetStateTester, checkSimpleTest) {
  std::vector<std::vector<double>> vec = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

  auto outer_dim = vec.size();
  auto inner_dim_0 = vec[0].size();
  auto inner_dim_1 = vec[1].size();
  auto inner_dim_2 = vec[2].size();

  std::cout << "outer_dim = " << outer_dim << "\n";
  std::cout << "inner_dim_0 = " << inner_dim_0 << "\n";
  std::cout << "inner_dim_1 = " << inner_dim_1 << "\n";
  std::cout << "inner_dim_2 = " << inner_dim_2 << "\n";
}