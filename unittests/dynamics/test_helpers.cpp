/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/helpers.h"
#include <gtest/gtest.h>
#include <vector>

using namespace cudaq;

TEST(OperatorHelpersTest, AggregateParameters_MultipleMappings) {
  std::vector<std::map<std::string, std::string>> mappings = {
      {{"alpha", "Parameter A"}, {"beta", "Parameter B"}},
      {{"alpha", "Updated Parameter A"}, {"gamma", "New Parameter"}}};

  auto result = OperatorHelpers::aggregate_parameters(mappings);

  EXPECT_EQ(result["alpha"], "Parameter A\n---\nUpdated Parameter A");
  EXPECT_EQ(result["beta"], "Parameter B");
  EXPECT_EQ(result["gamma"], "New Parameter");
}

TEST(OperatorHelpersTest, AggregateParameters_EmptyMappings) {
  std::vector<std::map<std::string, std::string>> mappings;
  auto result = OperatorHelpers::aggregate_parameters(mappings);

  EXPECT_TRUE(result.empty());
}

TEST(OperatorHelpersTest, ParameterDocs_ValidExtraction) {
  std::string docstring = "Description of function.\n"
                          "Arguments:\n"
                          "   alpha (float): The first parameter.\n"
                          "   beta (int): The second parameter.";

  auto result = OperatorHelpers::parameter_docs("alpha", docstring);
  EXPECT_EQ(result, "The first parameter.");

  result = OperatorHelpers::parameter_docs("beta", docstring);
  EXPECT_EQ(result, "The second parameter.");
}

TEST(OperatorHelpersTest, ParameterDocs_InvalidParam) {
  std::string docstring = "Description of function.\n"
                          "Arguments:\n"
                          "   alpha (float): The first parameter.\n"
                          "   beta (int): The second parameter.";

  auto result = OperatorHelpers::parameter_docs("gamma", docstring);
  EXPECT_EQ(result, "");
}

TEST(OperatorHelpersTest, ParameterDocs_EmptyDocString) {
  std::string docstring = "";
  auto result = OperatorHelpers::parameter_docs("alpha", docstring);
  EXPECT_EQ(result, "");
}

TEST(OperatorHelpersTest, GenerateAllStates_TwoQubits) {
  std::vector<int> degrees = {0, 1};
  std::map<int, int> dimensions = {{0, 2}, {1, 2}};

  auto states = OperatorHelpers::generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"00", "01", "10", "11"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_ThreeQubits) {
  std::vector<int> degrees = {0, 1, 2};
  std::map<int, int> dimensions = {{0, 2}, {1, 2}, {2, 2}};

  auto states = OperatorHelpers::generate_all_states(degrees, dimensions);
  std::vector<std::string> expected_states = {"000", "001", "010", "011",
                                              "100", "101", "110", "111"};

  EXPECT_EQ(states, expected_states);
}

TEST(OperatorHelpersTest, GenerateAllStates_EmptyDegrees) {
  std::vector<int> degrees;
  std::map<int, int> dimensions;

  auto states = OperatorHelpers::generate_all_states(degrees, dimensions);
  EXPECT_TRUE(states.empty());
}

TEST(OperatorHelpersTest, GenerateAllStates_MissingDegreesInMap) {
  std::vector<int> degrees = {0, 1, 2};
  std::map<int, int> dimensions = {{0, 2}, {1, 2}};

  EXPECT_THROW(OperatorHelpers::generate_all_states(degrees, dimensions),
               std::out_of_range);
}

TEST(OperatorHelpersTest, PermuteMatrix_SingleSwap) {
  Eigen::MatrixXcd matrix(2, 2);
  matrix << 1, 2, 3, 4;

  // Swap rows and columns
  std::vector<int> permutation = {1, 0};

  OperatorHelpers::permute_matrix(matrix, permutation);

  Eigen::MatrixXcd expected(2, 2);
  expected << 4, 3, 2, 1;

  EXPECT_EQ(matrix, expected);
}

TEST(OperatorHelpersTest, PermuteMatrix_IdentityPermutation) {
  Eigen::MatrixXcd matrix(3, 3);
  matrix << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  // No change
  std::vector<int> permutation = {0, 1, 2};

  OperatorHelpers::permute_matrix(matrix, permutation);

  Eigen::MatrixXcd expected(3, 3);
  expected << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  EXPECT_EQ(matrix, expected);
}

TEST(OperatorHelpersTest, CanonicalizeDegrees_SortedDescending) {
  std::vector<int> degrees = {3, 1, 2};
  auto sorted = OperatorHelpers::canonicalize_degrees(degrees);
  EXPECT_EQ(sorted, (std::vector<int>{3, 2, 1}));
}

TEST(OperatorHelpersTest, CanonicalizeDegrees_AlreadySorted) {
  std::vector<int> degrees = {5, 4, 3, 2, 1};
  auto sorted = OperatorHelpers::canonicalize_degrees(degrees);
  EXPECT_EQ(sorted, (std::vector<int>{5, 4, 3, 2, 1}));
}

TEST(OperatorHelpersTest, CanonicalizeDegrees_EmptyList) {
  std::vector<int> degrees;
  auto sorted = OperatorHelpers::canonicalize_degrees(degrees);
  EXPECT_TRUE(sorted.empty());
}

TEST(OperatorHelpersTest, ArgsFromKwargs_ValidArgs) {
  std::map<std::string, std::string> kwargs = {
      {"alpha", "0.5"}, {"beta", "1.0"}, {"gamma", "2.0"}};

  std::vector<std::string> required_args = {"alpha", "beta"};
  std::vector<std::string> kwonly_args = {"gamma"};

  auto [args, kwonly] =
      OperatorHelpers::args_from_kwargs(kwargs, required_args, kwonly_args);

  EXPECT_EQ(args.size(), 2);
  EXPECT_EQ(args[0], "0.5");
  EXPECT_EQ(args[1], "1.0");

  EXPECT_EQ(kwonly.size(), 1);
  EXPECT_EQ(kwonly["gamma"], "2.0");
}

TEST(OperatorHelpersTest, ArgsFromKwargs_MissingRequiredArgs) {
  std::map<std::string, std::string> kwargs = {{"beta", "1.0"},
                                               {"gamma", "2.0"}};

  std::vector<std::string> required_args = {"alpha", "beta"};
  std::vector<std::string> kwonly_args = {"gamma"};

  EXPECT_THROW(
      OperatorHelpers::args_from_kwargs(kwargs, required_args, kwonly_args),
      std::invalid_argument);
}
