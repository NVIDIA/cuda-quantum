/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.  * All rights reserved.
 *                                                      *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cmath>
#include <cudaq/algorithm.h>
#include <span>
#include <vector>

using namespace cudaq;

// Amplitude encoding builds a state vector; density-matrix backends differ.
#ifndef CUDAQ_BACKEND_STIM
#ifndef CUDAQ_BACKEND_DM

CUDAQ_TEST(EncodingTester, issueExample) {
  const std::vector<double> data{0.5, 0.5, 0.5};
  const state encoded = amplitude_encode(data);
  constexpr double kInvSqrt3 = 0.5773502691896258;

  EXPECT_NEAR(kInvSqrt3, encoded[0].real(), 1e-3);
  EXPECT_NEAR(kInvSqrt3, encoded[1].real(), 1e-3);
  EXPECT_NEAR(kInvSqrt3, encoded[2].real(), 1e-3);
  EXPECT_NEAR(0.0, encoded[3].real(), 1e-3);

  long double normSq = 0.0L;
  for (std::size_t i = 0; i < 4; ++i)
    normSq += std::norm(encoded[i]);
  EXPECT_NEAR(1.0, std::sqrt(static_cast<double>(normSq)), 1e-3);
}

CUDAQ_TEST(EncodingTester, alreadyPowerOfTwo) {
  const std::vector<double> data{1.0, 0.0, 0.0, 0.0};
  const state encoded = amplitude_encode(data);
  EXPECT_NEAR(1.0, encoded[0].real(), 1e-6);
  EXPECT_NEAR(0.0, encoded[1].real(), 1e-6);
  EXPECT_NEAR(0.0, encoded[2].real(), 1e-6);
  EXPECT_NEAR(0.0, encoded[3].real(), 1e-6);
}

CUDAQ_TEST(EncodingTester, stateRoundTrip) {
  const state first = amplitude_encode(std::vector<double>{0.5, 0.5, 0.5});
  const state again = amplitude_encode(first);
  for (std::size_t i = 0; i < 4; ++i)
    EXPECT_NEAR(first[i].real(), again[i].real(), 1e-6);
}

CUDAQ_TEST(EncodingTester, emptyInput) {
  EXPECT_THROW(amplitude_encode(std::span<const double>{}),
               std::invalid_argument);
}

CUDAQ_TEST(EncodingTester, zeroVector) {
  const std::vector<double> zeros{0.0, 0.0, 0.0};
  EXPECT_THROW(amplitude_encode(zeros), std::invalid_argument);
}

#endif // CUDAQ_BACKEND_DM
#endif // CUDAQ_BACKEND_STIM
