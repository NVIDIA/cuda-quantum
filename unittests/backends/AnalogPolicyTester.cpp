/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"
#include <gtest/gtest.h>

namespace {

// The launch path itself is covered by `targettests/analog`, which exercises it
// from an `nvq++`-built binary with a complete runtime.
TEST(AnalogPolicyTester, CompileTargetPreservesSourceModule) {
  cudaq::AnalogRemoteRESTQPU qpu;
  auto target = qpu.getCompileTarget();
  EXPECT_FALSE(target.overrideAOTCompilation);
}

} // namespace
