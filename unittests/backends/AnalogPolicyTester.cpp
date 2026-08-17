/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogRemoteRESTQPU.h"
#include "common/CompiledModule.h"
#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/platform.h"
#include <any>
#include <future>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

struct AnalogLaunchState {
  std::size_t syncLaunches = 0;
  std::size_t asyncLaunches = 0;
};

TEST(AnalogPolicyTester, CompileTargetPreservesSourceModule) {
  cudaq::AnalogRemoteRESTQPU qpu;
  auto target = qpu.getCompileTarget(cudaq::sample_policy{});
  EXPECT_FALSE(target.overrideAOTCompilation);
}

TEST(AnalogPolicyTester, RejectsUnexpectedKernelNames) {
  try {
    (void)cudaq::detail::launchAnalogKernel("unexpected", "{}", 1);
    FAIL() << "Expected launchAnalogKernel to reject the kernel name.";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(), "Unexpected type of kernel.");
  }

  try {
    (void)cudaq::detail::launchAnalogKernelAsync("unexpected", "{}", 1);
    FAIL() << "Expected launchAnalogKernelAsync to reject the kernel name.";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(), "Unexpected type of kernel.");
  }
}

TEST(AnalogPolicyTester, PreservesSourceModuleAndPayload) {
  AnalogLaunchState state;
  cudaq::RuntimeEndpoint endpoint{.impl = &state};
  endpoint.dispatch.set<cudaq::sample_policy>(
      +[](std::any &impl, const cudaq::sample_policy &policy,
          const cudaq::CompiledModule &module, cudaq::KernelArgs args) {
        auto *state = std::any_cast<AnalogLaunchState *>(impl);
        ++state->syncLaunches;
        EXPECT_EQ(policy.options.shots, 11);
        EXPECT_EQ(module.getName(), "__analog_hamiltonian_kernel__sync");
        auto functionPtr = module.getFunctionPtr();
        EXPECT_TRUE(functionPtr.has_value());
        if (functionPtr)
          EXPECT_EQ(functionPtr->getFn(), nullptr);
        auto packed = args.getPacked();
        EXPECT_TRUE(packed.has_value());
        if (packed) {
          auto *data = reinterpret_cast<const char *>(packed->data.data());
          EXPECT_EQ(std::string(data, packed->data.size()), R"({"sync":true})");
        }
        return cudaq::sample_result{};
      });
  endpoint.dispatch.set<cudaq::async_sample_policy>(
      +[](std::any &impl, const cudaq::async_sample_policy &policy,
          const cudaq::CompiledModule &module, cudaq::KernelArgs args) {
        auto *state = std::any_cast<AnalogLaunchState *>(impl);
        ++state->asyncLaunches;
        EXPECT_EQ(policy.inner.options.shots, 13);
        EXPECT_EQ(module.getName(), "__analog_hamiltonian_kernel__async");
        auto functionPtr = module.getFunctionPtr();
        EXPECT_TRUE(functionPtr.has_value());
        if (functionPtr)
          EXPECT_EQ(functionPtr->getFn(), nullptr);
        auto packed = args.getPacked();
        EXPECT_TRUE(packed.has_value());
        if (packed) {
          auto *data = reinterpret_cast<const char *>(packed->data.data());
          EXPECT_EQ(std::string(data, packed->data.size()),
                    R"({"async":true})");
        }
        std::promise<cudaq::sample_result> promise;
        promise.set_value({});
        return cudaq::async_sample_result(
            cudaq::detail::future(promise.get_future()));
      });

  cudaq::get_platform().setRuntimeEndpoint(std::move(endpoint));

  auto syncResult = cudaq::detail::launchAnalogKernel(
      "__analog_hamiltonian_kernel__sync", R"({"sync":true})", 11);
  auto asyncResult = cudaq::detail::launchAnalogKernelAsync(
      "__analog_hamiltonian_kernel__async", R"({"async":true})", 13);

  EXPECT_EQ(syncResult.get_total_shots(), 0);
  EXPECT_EQ(asyncResult.get().get_total_shots(), 0);
  EXPECT_EQ(state.syncLaunches, 1);
  EXPECT_EQ(state.asyncLaunches, 1);
}

} // namespace
