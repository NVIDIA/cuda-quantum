/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Unit tests for server-side observe helpers used by Fermioniq and external
/// custom QPU plugins.

#include "common/ObservableUserData.h"
#include "common/SampleResult.h"
#include "cudaq/operators.h"
#include <gtest/gtest.h>
#include <set>
#include <string>

TEST(ObservableUserDataTester, attachesFermioniqCompatibleObservable) {
  cudaq::KernelExecution code;
  auto spin =
      0.5 * cudaq::spin::z(0) + 0.3 * cudaq::spin::z(0) * cudaq::spin::z(1);

  cudaq::attachObservableUserData(code, spin);

  ASSERT_TRUE(code.user_data->contains("observable"));
  ASSERT_TRUE(code.user_data->at("observable").is_array());
  ASSERT_EQ(code.user_data->at("observable").size(), 2u);

  std::set<std::string> termIds;
  for (const auto &entry : code.user_data->at("observable")) {
    ASSERT_TRUE(entry.is_array());
    ASSERT_EQ(entry.size(), 2u);
    ASSERT_TRUE(entry[0].is_string());
    ASSERT_TRUE(entry[1].is_string());
    termIds.insert(entry[0].get<std::string>());
    // Coefficient strings match Fermioniq: "re±imj"
    const auto coeff = entry[1].get<std::string>();
    EXPECT_NE(coeff.find('j'), std::string::npos);
  }
  EXPECT_TRUE(termIds.count("Z0"));
  EXPECT_TRUE(termIds.count("Z0Z1"));
}

TEST(ObservableUserDataTester, serverSideExpectationUsesGlobalRegister) {
  // processResults for server-side observe backends return
  // ExecutionResult(double). async_observe_result::get() must accept
  // has_expectation() on the global register without requiring
  // spin_op->to_string() as the register name.
  cudaq::sample_result data(cudaq::ExecutionResult(/*expVal=*/0.41));
  EXPECT_TRUE(data.has_expectation());
  EXPECT_DOUBLE_EQ(data.expectation(), 0.41);
  EXPECT_TRUE(data.has_expectation(cudaq::GlobalRegisterName));
}
