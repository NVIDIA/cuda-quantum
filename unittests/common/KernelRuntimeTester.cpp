/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/host_config.h"
#include "cudaq/platform/KernelRuntime.h"
#include "cudaq/platform/nvqpp_interface.h"
#include <gtest/gtest.h>
#include <stdexcept>

extern "C" void
cudaqInitializeKernelRuntime(const cudaq::detail::KernelRuntimeCallbacks &);

namespace cudaq {
ExecutionManager *getExecutionManagerInternal();
ExecutionManager *getRegisteredExecutionManager();
namespace detail {
ExecutionManager *getExecutionManagerFromContext();
}
} // namespace cudaq

namespace {
int argument;
const std::vector<void *> rawArgs{&argument};
cudaq::KernelThunkResultType thunk(void *args, bool remote) {
  EXPECT_EQ(args, &argument);
  EXPECT_FALSE(remote);
  return {args, 23};
}

cudaq::KernelThunkResultType packedLaunch(const char *name,
                                          cudaq::KernelThunkType kernel,
                                          void *args, std::uint64_t size,
                                          std::uint64_t offset) {
  EXPECT_STREQ(name, "compiled_kernel");
  EXPECT_EQ(kernel, thunk);
  EXPECT_EQ(size, 42);
  EXPECT_EQ(offset, 17);
  return kernel(args, false);
}

class KernelRuntimeTest : public testing::Test {
  void TearDown() override { cudaqInitializeKernelRuntime({}); }
};

TEST_F(KernelRuntimeTest, RequiresRuntimeInitialization) {
  EXPECT_THROW((void)cudaq::altLaunchKernel("", nullptr, nullptr, 0, 0),
               std::runtime_error);
  EXPECT_THROW((void)cudaq::streamlinedLaunchKernel("", {}),
               std::runtime_error);
  EXPECT_THROW((void)cudaq::hybridLaunchKernel("", nullptr, nullptr, 0, 0, {}),
               std::runtime_error);
}

TEST_F(KernelRuntimeTest, PreservesLaunchArgumentsAndResults) {
  cudaq::detail::KernelRuntimeCallbacks callbacks{};
  callbacks.alt = packedLaunch;
  callbacks.streamlined = [](const char *name,
                             const std::vector<void *> &args) {
    EXPECT_STREQ(name, "compiled_kernel");
    EXPECT_EQ(&args, &rawArgs);
    return thunk(args[0], false);
  };
  callbacks.hybrid = [](const char *name, cudaq::KernelThunkType kernel,
                        void *args, std::uint64_t size, std::uint64_t offset,
                        const std::vector<void *> &raw) {
    EXPECT_EQ(&raw, &rawArgs);
    return packedLaunch(name, kernel, args, size, offset);
  };
  cudaqInitializeKernelRuntime(callbacks);

  for (auto result :
       {cudaq::altLaunchKernel("compiled_kernel", thunk, &argument, 42, 17),
        cudaq::streamlinedLaunchKernel("compiled_kernel", rawArgs),
        cudaq::hybridLaunchKernel("compiled_kernel", thunk, &argument, 42, 17,
                                  rawArgs)}) {
    EXPECT_EQ(result.data_buffer, &argument);
    EXPECT_EQ(result.size, 23);
  }
}

TEST_F(KernelRuntimeTest, ForwardsHostHeaderEntryPoints) {
  static int disableCalls = 0;
  static int backendCalls = 0;
  disableCalls = backendCalls = 0;
  auto *manager = reinterpret_cast<cudaq::ExecutionManager *>(&argument);
  cudaq::detail::KernelRuntimeCallbacks callbacks{};
  callbacks.executionManager = [] {
    return reinterpret_cast<cudaq::ExecutionManager *>(&argument);
  };
  callbacks.registeredExecutionManager = callbacks.executionManager;
  callbacks.contextExecutionManager = callbacks.executionManager;
  callbacks.disableYamlTargetConfig = [] { ++disableCalls; };
  callbacks.setTargetBackend = [](const char *backend) {
    EXPECT_STREQ(backend, "qpp-cpu");
    ++backendCalls;
  };
  cudaqInitializeKernelRuntime(callbacks);
  EXPECT_EQ(cudaq::getExecutionManagerInternal(), manager);
  EXPECT_EQ(cudaq::getRegisteredExecutionManager(), manager);
  EXPECT_EQ(cudaq::detail::getExecutionManagerFromContext(), manager);
  cudaq::detail::YamlTargetConfigDisabler disabler;
  cudaq::detail::TargetSetter setter("qpp-cpu");
  EXPECT_EQ(disableCalls, 1);
  EXPECT_EQ(backendCalls, 1);
}
} // namespace
