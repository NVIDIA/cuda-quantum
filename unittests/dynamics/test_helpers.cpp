/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/dynamics/helpers.h"
#include <gtest/gtest.h>
#include <vector>

using namespace cudaq::detail;

TEST(OperatorHelpersTest, GenerateAllStates_TwoQubits) {
  std::vector<int> degrees = {0, 1};
  std::unordered_map<int, int> dimensions = {{0, 2}, {1, 2}};

  auto states = generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"00", "01", "10", "11"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_ThreeQubits) {
  std::vector<int> degrees = {0, 1, 2};
  std::unordered_map<int, int> dimensions = {{0, 2}, {1, 2}, {2, 2}};

  auto states = generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"000", "001", "010", "011",
                                              "100", "101", "110", "111"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_EmptyDegrees) {
  std::vector<int> degrees;
  std::unordered_map<int, int> dimensions;

  auto states = generate_all_states(degrees, dimensions);
  EXPECT_TRUE(states.empty());
}

TEST(OperatorHelpersTest, PermuteMatrix_SingleSwap) {
  cudaq::matrix_2 matrix(2, 2);
  matrix[{0, 0}] = 1;
  matrix[{0, 1}] = 2;
  matrix[{1, 0}] = 3;
  matrix[{1, 1}] = 4;

  // Swap rows and columns
  std::vector<int> permutation = {1, 0};

  permute_matrix(matrix, permutation);

  cudaq::matrix_2 expected(2, 2);
  expected[{0, 0}] = 4;
  expected[{0, 1}] = 3;
  expected[{1, 0}] = 2;
  expected[{1, 1}] = 1;

  EXPECT_EQ(matrix, expected);
}

TEST(OperatorHelpersTest, PermuteMatrix_IdentityPermutation) {
  cudaq::matrix_2 matrix(3, 3);
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
  std::vector<int> permutation = {0, 1, 2};

  permute_matrix(matrix, permutation);

  EXPECT_EQ(matrix, matrix);
}
