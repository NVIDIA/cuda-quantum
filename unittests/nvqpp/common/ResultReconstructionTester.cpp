/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ResultReconstruction.h"
#include "nlohmann/json.hpp"
#include <gtest/gtest.h>

// These compact and partial cases enforce the default-sampling ordering
// convention, as validated in
// `test_explicit_measurements.py::test_measurement_order`:
// measured bits follow qubit allocation order, not `mz` execution order.
TEST(ResultReconstructionTester, ReordersCompactQirResults) {
  const nlohmann::json outputNames = {
      {{0, {2, "r2", 2}}, {1, {0, "r0", 0}}, {2, {1, "r1", 1}}}};
  const auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(outputNames);
  const auto result =
      cudaq::reconstructSampleResultFromResultIndexedMeasurements({{"110", 7}},
                                                                  resultMap);

  EXPECT_EQ(result.to_map(), cudaq::CountsDictionary({{"101", 7}}));
  EXPECT_EQ(result.to_map("r0"), cudaq::CountsDictionary({{"1", 7}}));
  EXPECT_EQ(result.to_map("r1"), cudaq::CountsDictionary({{"0", 7}}));
  EXPECT_EQ(result.to_map("r2"), cudaq::CountsDictionary({{"1", 7}}));
}

TEST(ResultReconstructionTester, ReordersPartialQirResults) {
  const nlohmann::json outputNames = {{{0, {2, "r2", 1}}, {1, {0, "r0", 0}}}};
  const auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(outputNames);
  const auto result =
      cudaq::reconstructSampleResultFromResultIndexedMeasurements({{"01", 5}},
                                                                  resultMap);

  EXPECT_EQ(result.to_map(), cudaq::CountsDictionary({{"10", 5}}));
}

TEST(ResultReconstructionTester, SupportsLegacyOutputNames) {
  const nlohmann::json outputNames = {{{0, {9, "r0"}}, {1, {5, "r1"}}}};
  const auto resultMap =
      cudaq::makeResultOutputMapFromEnrichedOutputNames(outputNames);
  const auto result =
      cudaq::reconstructSampleResultFromResultIndexedMeasurements({{"10", 2}},
                                                                  resultMap);

  EXPECT_EQ(result.to_map(), cudaq::CountsDictionary({{"10", 2}}));
}
