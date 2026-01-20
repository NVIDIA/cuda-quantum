/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithms/detectors.h>

#ifdef CUDAQ_BACKEND_STIM
CUDAQ_TEST(DetectorTester, checkEmpty) {
  auto kernel = []() __qpu__ {};
  auto detectorMzIndices = cudaq::detectors(kernel);
  EXPECT_EQ(detectorMzIndices.size(), 0);
}

CUDAQ_TEST(DetectorTester, checkSimple) {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);
    mz(q[0]);
    mz(q[1]);
    cudaq::detector(-2, -1);
  };
  const auto detectorMzIndices = cudaq::detectors(kernel);
  EXPECT_EQ(detectorMzIndices.size(), 1);
  EXPECT_EQ(detectorMzIndices[0], (std::vector<std::int64_t>{0, 1}));
}

CUDAQ_TEST(DetectorTester, checkSimple_stdvec) {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);
    mz(q[0]);
    mz(q[1]);
    cudaq::detector(std::vector<std::int64_t>{-2, -1});
  };
  const auto detectorMzIndices = cudaq::detectors(kernel);
  EXPECT_EQ(detectorMzIndices.size(), 1);
  EXPECT_EQ(detectorMzIndices[0], (std::vector<std::int64_t>{0, 1}));
}

CUDAQ_TEST(DetectorTester, checkSimple_loops) {
  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);
    mz(q[0]);
    mz(q[1]);
    for (int i = 0; i < 10; i++) {
      mz(q[0]);
      mz(q[1]);
      cudaq::detector(-4, -2);
      cudaq::detector(-3, -1);
    }
  };
  const auto detectorMzIndices = cudaq::detectors(kernel);
  EXPECT_EQ(detectorMzIndices.size(), 20);
  for (int i = 0; i < 20; i++) {
    // 2 measurements per detector
    EXPECT_EQ(detectorMzIndices[i].size(), 2);
    // E.g. 0,2 / 1,3 / 4,6 / 5,7 / ...
    EXPECT_EQ(detectorMzIndices[i][0], i);
    EXPECT_EQ(detectorMzIndices[i][1], i + 2);
  }
}
#endif
