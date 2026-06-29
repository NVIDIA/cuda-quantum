/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "AnyonHelper.h"
#include "CUDAQTestUtils.h"

#include <gtest/gtest.h>

// Two registers of equal length are concatenated per shot, in the order given.
CUDAQ_TEST(AnyonUtilTester, checkCombineRegisterResults) {
  const auto results = nlohmann::json::parse(R"(
    { "r0" : ["0", "1", "0"],
      "r1" : ["1", "1", "0"] })");
  auto bitstrings =
      cudaq::utils::anyon::combineRegisterResults(results, {"r0", "r1"});
  ASSERT_EQ(bitstrings.size(), 3);
  EXPECT_EQ(bitstrings[0], "01");
  EXPECT_EQ(bitstrings[1], "11");
  EXPECT_EQ(bitstrings[2], "00");
}

CUDAQ_TEST(AnyonUtilTester, checkRejectsUnequalRegisterLengths) {
  nlohmann::json results;
  results["r0"] = std::vector<std::string>{"0"};
  results["r1"] = std::vector<std::string>(64, "1");
  EXPECT_THROW(
      cudaq::utils::anyon::combineRegisterResults(results, {"r0", "r1"}),
      std::runtime_error);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
