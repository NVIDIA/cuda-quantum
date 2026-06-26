/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "IQMQubitMapping.h"
#include "nlohmann/json.hpp"
#include <functional>
#include <gtest/gtest.h>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

struct ArchitectureCase {
  std::string name;
  std::string architecture;
  // Expected dense device-qubit index per surviving provider name.
  std::map<std::string, uint> expectedNameMap;
  // Provider names expected to be dropped from the dense node set.
  std::vector<std::string> expectedDropped;
  // Expected cz adjacency over dense indices.
  std::vector<std::set<uint>> expectedAdjacency;
  // Expected dense-index-ordered provider name table.
  std::vector<std::string> expectedLabels;
};

} // namespace

// Each case feeds buildIqmArchitectureMapping a dynamic quantum architecture
// and asserts the dense device view it derives. The cases together exercise
// every branch of the node filter and the cz adjacency guard:
//   - isolated: every qubit supports prx and measure, so none is dropped and a
//     cz-partnerless qubit survives as an edge-free node.
//   - dropped: a qubit lacks measure, so it is erased from the dense node set
//     and the cz locus referencing it is excluded from the adjacency map.
static std::vector<ArchitectureCase> architectureCases() {
  return {{"isolated",
           R"({
         "qubits": ["QB1", "QB2", "QB3", "QB4"],
         "gates": {
           "cz": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1", "QB2"], ["QB2", "QB3"]]}
             }
           },
           "prx": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1"], ["QB2"], ["QB3"], ["QB4"]]}
             }
           },
           "measure": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1"], ["QB2"], ["QB3"], ["QB4"]]}
             }
           }
         }
       })",
           {{"QB1", 0u}, {"QB2", 1u}, {"QB3", 2u}, {"QB4", 3u}},
           {},
           {{1}, {0, 2}, {1}, {}},
           {"QB1", "QB2", "QB3", "QB4"}},
          {"dropped",
           R"({
         "qubits": ["QB1", "QB2", "QB3"],
         "gates": {
           "cz": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1", "QB2"], ["QB2", "QB3"]]}
             }
           },
           "prx": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1"], ["QB2"], ["QB3"]]}
             }
           },
           "measure": {
             "default_implementation": "impl",
             "implementations": {
               "impl": {"loci": [["QB1"], ["QB2"]]}
             }
           }
         }
       })",
           {{"QB1", 0u}, {"QB2", 1u}},
           {"QB3"},
           {{1}, {0}},
           {"QB1", "QB2"}}};
}

TEST(IQMQubitMappingTester, DerivesDenseDeviceViewFromArchitecture) {
  for (const auto &testCase : architectureCases()) {
    SCOPED_TRACE(testCase.name);
    auto mapping = cudaq::buildIqmArchitectureMapping(
        nlohmann::json::parse(testCase.architecture));

    ASSERT_EQ(mapping.qubitNameMap.size(), testCase.expectedNameMap.size());
    for (const auto &[name, index] : testCase.expectedNameMap)
      EXPECT_EQ(mapping.qubitNameMap.at(name), index);
    for (const auto &name : testCase.expectedDropped)
      EXPECT_EQ(mapping.qubitNameMap.count(name), 0u);

    ASSERT_EQ(mapping.qubitAdjacencyMap.size(),
              testCase.expectedAdjacency.size());
    for (std::size_t i = 0; i < testCase.expectedAdjacency.size(); ++i)
      EXPECT_EQ(mapping.qubitAdjacencyMap[i], testCase.expectedAdjacency[i]);

    EXPECT_EQ(mapping.backendLabels, testCase.expectedLabels);
  }
}

// buildIqmQubitMapping has two production inputs. When the mapper produced a
// target qubit mapping for an emitted execution, the mapping is composed with
// the base label table. When there is no target mapping, the full
// dynamic-architecture name map is used directly.
TEST(IQMQubitMappingTester, BuildsMappingFromActiveQubitsOrFullNameMap) {
  std::vector<std::string> backendLabels = {"QB1", "QB2", "QB3", "QB4"};
  auto labelTable = [&](cudaq::DeviceQubit q) -> std::optional<std::string> {
    if (q >= backendLabels.size() || backendLabels[q].empty())
      return std::nullopt;
    return backendLabels[q];
  };
  auto noLabels = [](cudaq::DeviceQubit) -> std::optional<std::string> {
    return std::nullopt;
  };
  std::map<std::string, uint, cudaq::IqmQubitOrder> emptyNameMap;
  auto architectureNameMap =
      cudaq::buildIqmArchitectureMapping(
          nlohmann::json::parse(architectureCases().front().architecture))
          .qubitNameMap;

  struct Expected {
    std::string logical;
    std::string physical;
  };
  struct Case {
    std::string name;
    cudaq::TargetQubitMapping targetQubitMapping;
    std::map<std::string, uint, cudaq::IqmQubitOrder> nameMap;
    std::function<std::optional<std::string>(cudaq::DeviceQubit)> backendLabel;
    std::vector<Expected> expected;
  };

  std::vector<Case> cases = {
      {"partialFromTargetMappingAndLabelTable",
       {{"QB4", 3}, {"QB2", 1}},
       emptyNameMap,
       labelTable,
       {{"QB4", "QB4"}, {"QB2", "QB2"}}},
      {"partialFromDenseNamesWithoutLabelTable",
       {{"QB1", 0}, {"QB3", 2}},
       emptyNameMap,
       noLabels,
       {{"QB1", "QB1"}, {"QB3", "QB3"}}},
      {"fullFromNameMapWhenNoActiveQubits",
       {},
       architectureNameMap,
       noLabels,
       {{"QB1", "QB1"}, {"QB2", "QB2"}, {"QB3", "QB3"}, {"QB4", "QB4"}}}};

  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    auto qubitMapping = cudaq::buildIqmQubitMapping(
        testCase.targetQubitMapping, testCase.nameMap, testCase.backendLabel);

    ASSERT_EQ(qubitMapping.size(), testCase.expected.size());
    for (std::size_t i = 0; i < testCase.expected.size(); ++i) {
      EXPECT_EQ(qubitMapping[i]["logical_name"], testCase.expected[i].logical);
      EXPECT_EQ(qubitMapping[i]["physical_name"],
                testCase.expected[i].physical);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
