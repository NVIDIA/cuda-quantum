/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <dlfcn.h>

#include <filesystem>
#include <string>

namespace {

void expectEmbeddedModule(const char *libraryPath) {
  void *const library = dlopen(libraryPath, RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(library, nullptr) << dlerror();
  void *const module = dlsym(library, "custatevecExCommunicatorGetModuleEXT");
  ASSERT_NE(module, nullptr) << dlerror();

  Dl_info moduleInfo{};
  ASSERT_NE(dladdr(module, &moduleInfo), 0);
  EXPECT_EQ(std::filesystem::canonical(moduleInfo.dli_fname),
            std::filesystem::canonical(libraryPath));
  EXPECT_EQ(dlclose(library), 0);
}

} // namespace

TEST(CuStateVecCommunicatorPackaging, ModuleIsEmbeddedInMgpuPlugins) {
  expectEmbeddedModule(CUSTATEVEC_MGPU_FP32_PATH);
  expectEmbeddedModule(CUSTATEVEC_MGPU_FP64_PATH);
  EXPECT_FALSE(std::filesystem::exists(CUSTATEVEC_STANDALONE_COMM_PATH));
}
