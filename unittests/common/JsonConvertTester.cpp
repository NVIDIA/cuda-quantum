/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/JsonConvert.h"
#include "nvqir/CircuitSimulator.h"
#include <gtest/gtest.h>

TEST(JsonConvertTester, CreateFromSizeAndPtrNullPtr) {
  auto *sim = cudaq::get_simulator();
  ASSERT_NE(sim, nullptr);
  cudaq::state_data data =
      std::make_pair(static_cast<std::complex<double> *>(nullptr),
                     static_cast<std::size_t>(4));
  EXPECT_ANY_THROW(sim->createStateFromData(data));
}

TEST(JsonConvertTester, CreateFromSizeAndPtrZeroSize) {
  auto *sim = cudaq::get_simulator();
  ASSERT_NE(sim, nullptr);
  std::complex<double> dummy{1.0, 0.0};
  cudaq::state_data data = std::make_pair(&dummy, static_cast<std::size_t>(0));
  EXPECT_ANY_THROW(sim->createStateFromData(data));
}

TEST(JsonConvertTester, CreateFromSizeAndPtrValid) {
  auto *sim = cudaq::get_simulator();
  ASSERT_NE(sim, nullptr);
  std::vector<std::complex<double>> stateVec = {{1.0, 0.0}, {0.0, 0.0}};
  cudaq::state_data data = std::make_pair(stateVec.data(), stateVec.size());
  std::unique_ptr<cudaq::SimulationState> state;
  EXPECT_NO_THROW(state = sim->createStateFromData(data));
  ASSERT_NE(state, nullptr);
  EXPECT_EQ(state->getNumQubits(), 1);
  EXPECT_EQ(state->getNumElements(), 2u);
}

namespace {
/// Build a "minimal" `ExecutionContext` JSON with the given `simulationData`
static json makeExecContextJson(const json &simData = json()) {
  json j;
  j["name"] = "extract-state";
  j["shots"] = -1;
  j["hasConditionalsOnMeasureResults"] = false;
  j["result"] = json::array();
  j["registerNames"] = json::array();
  if (!simData.is_null())
    j["simulationData"] = simData;
  return j;
}
} // namespace

TEST(JsonConvertTester, SimDataDimMismatch0) {
  json simData;
  simData["dim"] = {1024};
  simData["data"] = {{1.0, 0.0}};
  json j = makeExecContextJson(simData);
  cudaq::ExecutionContext ctx("extract-state");
  EXPECT_ANY_THROW(cudaq::from_json(j, ctx));
}

TEST(JsonConvertTester, SimDataDimMismatch1) {
  json simData;
  simData["dim"] = {4};
  simData["data"] = json::array();
  json j = makeExecContextJson(simData);
  cudaq::ExecutionContext ctx("extract-state");
  EXPECT_ANY_THROW(cudaq::from_json(j, ctx));
}

TEST(JsonConvertTester, SimDataDimMismatch2) {
  json simData;
  simData["dim"] = json::array();
  simData["data"] = {{1.0, 0.0}};
  json j = makeExecContextJson(simData);
  cudaq::ExecutionContext ctx("extract-state");
  EXPECT_ANY_THROW(cudaq::from_json(j, ctx));
}
