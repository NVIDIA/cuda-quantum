/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/algorithms/policy_dispatch.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/PTSBESampler.h"

using namespace cudaq;

const ptsbe::PTSBETrace kXTrace = {
    {ptsbe::TraceInstructionType::Gate, "x", {0}, {}, {}}};

CUDAQ_TEST(PTSBEPolicyTest, RegistryDispatch) {
  auto kind = [](auto policy) -> std::string {
    using Policy = std::decay_t<decltype(policy)>;
    if constexpr (std::is_same_v<Policy, ptsbe_sample_policy>)
      return "ptsbe";
    else if constexpr (std::is_same_v<Policy, other_policies>)
      return "other";
    else
      return "unexpected";
  };

  EXPECT_EQ(policies::withPolicy(ptsbe_sample_policy::name, kind), "ptsbe");
  EXPECT_EQ(policies::withPolicy("no-such-policy", kind), "other");
}

CUDAQ_TEST(PTSBEPolicyTest, PolicyName) {
  EXPECT_STREQ(ptsbe_sample_policy::name, "ptsbe-sample");
}

CUDAQ_TEST(PTSBEPolicyTest, ExecuteBatchAggregatesResults) {
  ptsbe::PTSBatch batch;
  batch.trace = kXTrace;
  batch.measureQubits = {0};
  batch.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{}, 0.7,
                                  7);
  batch.trajectories.emplace_back(1, std::vector<cudaq::KrausSelection>{}, 0.3,
                                  3);

  ptsbe_sample_policy policy;
  policy.batch = &batch;
  policy.shots = batch.totalShots();

  auto result = ptsbe::detail::executeBatch(policy);

  EXPECT_EQ(result.count("1"), 10);
  EXPECT_EQ(result.count("0"), 0);
  ASSERT_EQ(policy.perTrajectoryResults.size(), 2);
  EXPECT_EQ(policy.perTrajectoryResults[0].count("1"), 7);
  EXPECT_EQ(policy.perTrajectoryResults[1].count("1"), 3);

  EXPECT_EQ(cudaq::getExecutionContext(), nullptr);
}

CUDAQ_TEST(PTSBEPolicyTest, FinalizeWithoutBatchThrows) {
  ptsbe_sample_policy policy;
  EXPECT_THROW(ptsbe::detail::finalizePTSBE(policy), std::runtime_error);
}

CUDAQ_TEST(PTSBEPolicyTest, ContextRestoredAfterFailure) {
  ptsbe::PTSBatch badBatch;
  badBatch.trace = {
      {ptsbe::TraceInstructionType::Gate, "not_a_gate", {0}, {}, {}}};
  badBatch.measureQubits = {0};
  badBatch.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{},
                                     1.0, 5);

  ptsbe_sample_policy policy;
  policy.batch = &badBatch;
  policy.shots = badBatch.totalShots();

  EXPECT_ANY_THROW(ptsbe::detail::executeBatch(policy));
  EXPECT_EQ(cudaq::getExecutionContext(), nullptr);

  ptsbe::PTSBatch goodBatch;
  goodBatch.trace = kXTrace;
  goodBatch.measureQubits = {0};
  goodBatch.trajectories.emplace_back(0, std::vector<cudaq::KrausSelection>{},
                                      1.0, 5);

  auto results = ptsbe::detail::samplePTSBEWithLifecycle(goodBatch);
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0].count("1"), 5);
  EXPECT_EQ(cudaq::getExecutionContext(), nullptr);
}