/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RuntimeTarget.h"
#include <gtest/gtest.h>

TEST(RuntimeTargetTester, defaultServerHelperLibDirIsEmpty) {
  cudaq::RuntimeTarget target;
  EXPECT_TRUE(target.serverHelperLibDir.empty());
}

TEST(RuntimeTargetTester, serverHelperLibDirCanBeSet) {
  cudaq::RuntimeTarget target;
  target.serverHelperLibDir = "/opt/my-backend/lib";
  EXPECT_EQ(target.serverHelperLibDir, "/opt/my-backend/lib");
}

TEST(RuntimeTargetTester, serverHelperLibDirIsIndependentOfName) {
  cudaq::RuntimeTarget target;
  target.name = "my-backend";
  target.serverHelperLibDir = "/opt/my-backend/lib";
  EXPECT_EQ(target.name, "my-backend");
  EXPECT_EQ(target.serverHelperLibDir, "/opt/my-backend/lib");
}
