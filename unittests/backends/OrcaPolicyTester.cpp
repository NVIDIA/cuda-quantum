/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompiledModule.h"
#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/algorithms/policy_dispatch.h"
#include "cudaq/orca.h"
#include "cudaq/platform.h"
#include "cudaq/platform/orca/OrcaRemoteRESTQPU.h"
#include <any>
#include <future>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct OrcaLaunchState {
  std::size_t syncLaunches = 0;
  std::size_t asyncLaunches = 0;
};

TEST(OrcaPolicyTester, CompileTargetPreservesSourceModule) {
  cudaq::OrcaRemoteRESTQPU qpu;
  auto target = qpu.getCompileTarget();
  EXPECT_FALSE(target.overrideAOTCompilation);
}

TEST(OrcaPolicyTester, RegistryDispatch) {
  auto kind = [](auto policy) -> std::string {
    using Policy = std::decay_t<decltype(policy)>;
    if constexpr (std::is_same_v<Policy, cudaq::orca::sample_policy>)
      return "orca";
    else
      return "unexpected";
  };

  EXPECT_EQ(cudaq::policies::withPolicy(cudaq::orca::sample_policy::name, kind),
            "orca");
}

TEST(OrcaPolicyTester, PublicApisReturnEndpointResults) {
  OrcaLaunchState state;
  cudaq::RuntimeEndpoint endpoint{.impl = &state};
  endpoint.dispatch.set<cudaq::orca::sample_policy>(
      +[](std::any &impl, const cudaq::orca::sample_policy &,
          const cudaq::CompiledModule &module, cudaq::KernelArgs args) {
        auto *state = std::any_cast<OrcaLaunchState *>(impl);
        ++state->syncLaunches;
        EXPECT_EQ(module.getName(), cudaq::orca::sample_policy::kernelName);
        auto functionPtr = module.getFunctionPtr();
        EXPECT_TRUE(functionPtr.has_value());
        if (functionPtr)
          EXPECT_EQ(functionPtr->getFn(), nullptr);
        auto packed = args.getPacked();
        EXPECT_TRUE(packed.has_value());
        if (packed) {
          EXPECT_GE(packed->data.size(), sizeof(cudaq::orca::TBIParameters));
          if (packed->data.size() >= sizeof(cudaq::orca::TBIParameters)) {
            auto *parameters =
                reinterpret_cast<const cudaq::orca::TBIParameters *>(
                    packed->data.data());
            EXPECT_EQ(parameters->n_samples, 5);
          }
        }
        return cudaq::sample_result{};
      });
  endpoint.dispatch.set<cudaq::orca::async_sample_policy>(
      +[](std::any &impl, const cudaq::orca::async_sample_policy &,
          const cudaq::CompiledModule &module, cudaq::KernelArgs args) {
        auto *state = std::any_cast<OrcaLaunchState *>(impl);
        ++state->asyncLaunches;
        EXPECT_EQ(module.getName(), cudaq::orca::sample_policy::kernelName);
        auto functionPtr = module.getFunctionPtr();
        EXPECT_TRUE(functionPtr.has_value());
        if (functionPtr)
          EXPECT_EQ(functionPtr->getFn(), nullptr);
        auto packed = args.getPacked();
        EXPECT_TRUE(packed.has_value());
        if (packed) {
          EXPECT_GE(packed->data.size(), sizeof(cudaq::orca::TBIParameters));
          if (packed->data.size() >= sizeof(cudaq::orca::TBIParameters)) {
            auto *parameters =
                reinterpret_cast<const cudaq::orca::TBIParameters *>(
                    packed->data.data());
            EXPECT_EQ(parameters->n_samples, 7);
          }
        }
        std::promise<cudaq::sample_result> promise;
        promise.set_value({});
        return cudaq::async_sample_result(
            cudaq::detail::future(promise.get_future()));
      });

  cudaq::get_platform().setRuntimeEndpoint(std::move(endpoint));

  std::vector<std::size_t> inputState{1};
  std::vector<std::size_t> loopLengths{1};
  std::vector<double> beamSplitterAngles{0.1};
  std::vector<double> phaseShifterAngles{0.2};

  auto syncResult =
      cudaq::orca::sample(inputState, loopLengths, beamSplitterAngles,
                          phaseShifterAngles, /*n_samples=*/5);
  auto asyncResult =
      cudaq::orca::sample_async(inputState, loopLengths, beamSplitterAngles,
                                phaseShifterAngles, /*n_samples=*/7);

  EXPECT_EQ(syncResult.get_total_shots(), 0);
  EXPECT_EQ(asyncResult.get().get_total_shots(), 0);
  EXPECT_EQ(state.syncLaunches, 1);
  EXPECT_EQ(state.asyncLaunches, 1);
}

} // namespace
