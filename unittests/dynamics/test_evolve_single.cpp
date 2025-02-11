// /*******************************************************************************
//  * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates. *
//  * All rights reserved. *
//  * *
//  * This source code and the accompanying materials are made available under *
//  * the terms of the Apache License 2.0 which accompanies this distribution. *
//  ******************************************************************************/

#include "cudaq/evolution.h"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>

TEST(EvolveTester, checkSimple) {
  const std::map<int, int> dims = {{0, 2}};
  const std::string op_id = "pauli_x";
  auto func = [](std::vector<int> dimensions,
                 std::map<std::string, std::complex<double>> _none) {
    if (dimensions.size() != 1)
      throw std::invalid_argument("Must have a singe dimension");
    if (dimensions[0] != 2)
      throw std::invalid_argument("Must have dimension 2");
    auto mat = cudaq::matrix_2(2, 2);
    mat[{1, 0}] = 1.0;
    mat[{0, 1}] = 1.0;
    return mat;
  };
  cudaq::matrix_operator::define(op_id, {-1}, func);
  auto ham = cudaq::product_operator<cudaq::matrix_operator>(
      std::complex<double>{0.0, -1.0} * 2.0 * M_PI * 0.1,
      cudaq::matrix_operator(op_id, {0}));
  cudaq::Schedule schedule({0.0, 0.1, 0.2});
  auto initialState =
      cudaq::state::from_data(std::vector<std::complex<double>>{1.0, 0.0});
  auto result =
      cudaq::evolve_single(ham, dims, schedule, initialState, {}, {&ham});
}
