/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RuntimeTarget.h"
#include <filesystem>
#include <gtest/gtest.h>

TEST(RuntimeTargetTester, defaultPluginLibDirIsEmpty) {
  cudaq::RuntimeTarget target;
  EXPECT_TRUE(target.pluginLibDir.empty());
}

TEST(RuntimeTargetTester, pluginLibDirCanBeSet) {
  cudaq::RuntimeTarget target;
  target.pluginLibDir = "/opt/my-backend/lib";
  EXPECT_EQ(target.pluginLibDir, "/opt/my-backend/lib");
}

TEST(RuntimeTargetTester, pluginLibDirIsIndependentOfName) {
  cudaq::RuntimeTarget target;
  target.name = "my-backend";
  target.pluginLibDir = "/opt/my-backend/lib";
  EXPECT_EQ(target.name, "my-backend");
  EXPECT_EQ(target.pluginLibDir, "/opt/my-backend/lib");
}

TEST(RuntimeTargetTester, configPath_emptyByDefault) {
  cudaq::RuntimeTarget target;
  EXPECT_TRUE(target.configPath.empty());
}

TEST(RuntimeTargetTester, configPathCanBeSet) {
  cudaq::RuntimeTarget target;
  target.configPath = "/opt/my-backend/targets/my-backend.so";
  EXPECT_EQ(target.configPath,
            std::filesystem::path("/opt/my-backend/targets/my-backend.so"));
}
