/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.  *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ResultReconstruction.h"
#include <gtest/gtest.h>
#include <stdexcept>

// The happy-path reconstruction behavior (bit-index projection, sparse gaps,
// named-register placement, observe-split, per-shot sequential data, and
// legacy two-tuple fallback) is exercised end-to-end against the real
// emitKernelExecutions path in runtime/test/test_kernel_execution_maps.cpp.
// The ServerHelper bit-index path
// (resultMapForJob/tryReconstructFromDeviceIndexedCounts) is covered through a
// real backend in unittests/nvqpp/backends/iqm/IQMTester.cpp. The case below
// covers the one behavior neither exercises: guards against malformed provider
// bitstrings, which arrive only from external backend responses and are never
// produced by the internal codegen.
TEST(ResultReconstructionTester, ReconstructsCompactProviderCounts) {
  cudaq::ResultOutputMap resultMap;
  resultMap.outputs = {
      {.resultIndex = 0,
       .deviceQubit = 5,
       .outputName = "alpha",
       .outputPosition = 0},
      {.resultIndex = 1,
       .deviceQubit = 9,
       .outputName = "alpha",
       .outputPosition = 1},
  };

  auto result = cudaq::reconstructSampleResultFromResultIndexedMeasurements(
      {{"01", 3}, {"10", 2}}, resultMap);

  EXPECT_EQ(result.to_map().at("01"), 3);
  EXPECT_EQ(result.to_map().at("10"), 2);
  EXPECT_EQ(result.to_map("alpha").at("01"), 3);
  EXPECT_EQ(result.to_map("alpha").at("10"), 2);
}

TEST(ResultReconstructionTester, RejectsMalformedDeviceBitstrings) {
  struct Case {
    const char *name;
    std::vector<cudaq::ResultOutputEntry> outputs;
    cudaq::CountsDictionary counts;
  };

  std::vector<Case> cases = {
      {"bitstringShorterThanMappedBitIndex",
       {{.resultIndex = 0,
         .deviceQubit = 3,
         .outputName = "alpha",
         .outputPosition = 0}},
       {{"10", 1}}},
      {"invalidReturnedBits",
       {{.resultIndex = 0,
         .deviceQubit = 0,
         .outputName = "alpha",
         .outputPosition = 0}},
       {{"2", 1}}},
  };

  for (const auto &testCase : cases) {
    SCOPED_TRACE(testCase.name);
    cudaq::ResultOutputMap resultMap;
    resultMap.outputs = testCase.outputs;
    EXPECT_THROW(cudaq::reconstructSampleResultFromDeviceIndexedMeasurements(
                     testCase.counts, resultMap),
                 std::invalid_argument);
  }
}
