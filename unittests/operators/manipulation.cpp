/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/operators.h"
#include "cudaq/operators/helpers.h"
#include <gtest/gtest.h>

using namespace cudaq::detail;

TEST(OperatorHelpersTest, GenerateAllStates_TwoQubits) {
  std::vector<std::size_t> degrees = {0, 1};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}};

  auto states = generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"00", "10", "01", "11"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_ThreeQubits) {
  std::vector<std::size_t> degrees = {0, 1, 2};
  cudaq::dimension_map dimensions = {{0, 2}, {1, 2}, {2, 2}};

  auto states = generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"000", "100", "010", "110",
                                              "001", "101", "011", "111"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_EmptyDegrees) {
  std::vector<std::size_t> degrees;
  cudaq::dimension_map dimensions;

  auto states = generate_all_states(degrees, dimensions);
  EXPECT_TRUE(states.empty());
}

TEST(OperatorHelpersTest, PermuteMatrix_SingleSwap) {
  cudaq::complex_matrix matrix(2, 2);
  matrix[{0, 0}] = 1;
  matrix[{0, 1}] = 2;
  matrix[{1, 0}] = 3;
  matrix[{1, 1}] = 4;

  // Swap rows and columns
  std::vector<std::size_t> permutation = {1, 0};

  permute_matrix(matrix, permutation);

  cudaq::complex_matrix expected(2, 2);
  expected[{0, 0}] = 4;
  expected[{0, 1}] = 3;
  expected[{1, 0}] = 2;
  expected[{1, 1}] = 1;

  EXPECT_EQ(matrix, expected);
}

TEST(OperatorHelpersTest, PermuteMatrix_IdentityPermutation) {
  cudaq::complex_matrix matrix(3, 3);
  matrix[{0, 0}] = 1;
  matrix[{0, 1}] = 2;
  matrix[{0, 2}] = 3;
  matrix[{1, 0}] = 4;
  matrix[{1, 1}] = 5;
  matrix[{1, 2}] = 6;
  matrix[{2, 0}] = 7;
  matrix[{2, 1}] = 8;
  matrix[{2, 2}] = 9;

  // No change
  std::vector<std::size_t> permutation = {0, 1, 2};

  permute_matrix(matrix, permutation);

  EXPECT_EQ(matrix, matrix);
}
