/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompiledModule.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Target/RuntimeEndpoint.h"
#include "cudaq/algorithms/dem/policy.h"
#include "cudaq/algorithms/msm/policy.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/algorithms/resource_estimation.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/ptsbe/policy.h"
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace cudaq;

namespace {

class CompileTargetTestQPU : public QPU {
public:
  /// Number of times `launchKernel(sample_policy)` was called on this QPU.
  std::size_t sampleLaunchCount = 0;

  std::unique_ptr<CompileTarget>
  getCompileTarget(const sample_policy &) override {
    auto ct = std::make_unique<CompileTarget>();
    ct->emitJit = true;
    ct->fullySpecialize = false;
    ct->overrideAOTCompilation = true;
    return ct;
  }

  std::unique_ptr<CompileTarget> getCompileTarget(const other_policies &,
                                                  ExecutionContext *) override {
    auto ct = std::make_unique<CompileTarget>();
    ct->emitResourceCounts = true;
    ct->emitJit = false;
    ct->fullySpecialize = false;
    return ct;
  }

  sample_result launchKernel(const sample_policy &, const CompiledModule &,
                             KernelArgs) override {
    ++sampleLaunchCount;
    return {};
  }

  void enqueue(QuantumTask &) override {}
  KernelThunkResultType unifiedLaunchModule(const AnyModule &,
                                            KernelArgs) override {
    return {};
  }
};

class TestPlatform : public quantum_platform {
public:
  using quantum_platform::setCompileTarget;
  using quantum_platform::setRuntimeEndpoint;

  explicit TestPlatform(std::size_t numQpus = 1) {
    for (std::size_t i = 0; i < numQpus; ++i)
      platformQPUs.emplace_back(std::make_unique<CompileTargetTestQPU>());
  }

  CompileTargetTestQPU *getQpu(std::size_t qpuId) {
    return static_cast<CompileTargetTestQPU *>(platformQPUs[qpuId].get());
  }
};

CompileTarget makePlatformCompileTarget() {
  CompileTarget ct;
  ct.emitJit = false;
  ct.fullySpecialize = true;
  ct.overrideAOTCompilation = false;
  ct.supportDeviceCalls = true;
  return ct;
}

sample_result taggedSampleFn(std::any &impl, const sample_policy &,
                             const CompiledModule &, KernelArgs) {
  EXPECT_EQ(std::any_cast<int>(impl), 99);
  return {};
}

/// A launch function that counts its invocations in the endpoint state, to
/// check that writes to `impl` stick across launches.
sample_result countingSampleFn(std::any &impl, const sample_policy &,
                               const CompiledModule &, KernelArgs) {
  impl = std::any_cast<int>(impl) + 1;
  return {};
}

template <typename Policy>
void setLaunchFn(RuntimeEndpoint &endpoint, detail::launch_fn_type<Policy> fn) {
  endpoint.*detail::runtime_endpoint_fn<Policy>::member = fn;
}

template <typename Policy>
typename Policy::result_type
recordDispatchFn(std::any &impl, const Policy &policy, const CompiledModule &,
                 KernelArgs) {
  EXPECT_EQ(std::any_cast<int>(impl), 42);
  impl = get_policy_name(policy);
  return typename Policy::result_type{};
}

template <typename Policy>
void testPolicyDispatch(const Policy &policy) {
  RuntimeEndpoint endpoint{.impl = 42};
  setLaunchFn<Policy>(endpoint, recordDispatchFn<Policy>);

  CompiledModule module;
  (void)endpoint.launchKernel(policy, module, {});

  EXPECT_EQ(std::any_cast<std::string>(endpoint.impl), get_policy_name(policy));
}

template <typename Fn>
void expectOverrideDisabled(Fn &&fn, const std::string &what) {
  try {
    fn();
    FAIL() << "expected std::runtime_error for '" << what << "'";
  } catch (const std::runtime_error &e) {
    std::string msg = e.what();
    EXPECT_NE(msg.find(what), std::string::npos) << msg;
    EXPECT_NE(msg.find("manually setting a runtime endpoint"),
              std::string::npos)
        << msg;
  }
}

} // namespace

TEST(QuantumPlatformCompileTargetTester, fallsBackToQpuWhenUnset) {
  TestPlatform platform;
  sample_policy policy{.kernelName = "test_kernel"};

  auto ct = platform.getCompileTarget(policy);
  ASSERT_NE(ct, nullptr);
  EXPECT_TRUE(ct->emitJit);
  EXPECT_FALSE(ct->fullySpecialize);
  EXPECT_TRUE(ct->overrideAOTCompilation);
}

TEST(QuantumPlatformCompileTargetTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  sample_policy policy{.kernelName = "test_kernel"};

  auto ct = platform.getCompileTarget(policy);
  ASSERT_NE(ct, nullptr);
  EXPECT_FALSE(ct->emitJit);
  EXPECT_TRUE(ct->fullySpecialize);
  EXPECT_FALSE(ct->overrideAOTCompilation);
  EXPECT_TRUE(ct->supportDeviceCalls);
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesFallsBackToQpuWhenUnset) {
  TestPlatform platform;
  other_policies policy;

  auto ct = platform.getCompileTarget(policy);
  ASSERT_NE(ct, nullptr);
  EXPECT_TRUE(ct->emitResourceCounts);
  EXPECT_FALSE(ct->emitJit);
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesUsesPlatformOverride) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  other_policies policy;

  auto ct = platform.getCompileTarget(policy);
  ASSERT_NE(ct, nullptr);
  EXPECT_FALSE(ct->emitJit);
  EXPECT_TRUE(ct->fullySpecialize);
  EXPECT_FALSE(ct->emitResourceCounts);
}

TEST(QuantumPlatformCompileTargetTester, rejectsInvalidQpuId) {
  TestPlatform platform;
  sample_policy policy{.kernelName = "test_kernel"};

  EXPECT_THROW((void)platform.getCompileTarget(policy, /*qpu_id=*/1),
               std::invalid_argument);
}

TEST(QuantumPlatformCompileTargetTester, clearingOverrideFallsBackToQpu) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  sample_policy policy{.kernelName = "test_kernel"};

  platform.setCompileTarget(std::nullopt);

  auto ct = platform.getCompileTarget(policy);
  ASSERT_NE(ct, nullptr);
  EXPECT_TRUE(ct->emitJit);
  EXPECT_TRUE(ct->overrideAOTCompilation);
}

TEST(QuantumPlatformRuntimeEndpointTester, fallsBackToQpuWhenUnset) {
  TestPlatform platform;
  auto &endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);

  auto *qpu = std::any_cast<QPU *>(endpoint.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(0));
  EXPECT_NE(endpoint.sample, nullptr);
}

TEST(QuantumPlatformRuntimeEndpointTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(
      RuntimeEndpoint{.sample = taggedSampleFn, .impl = 42});

  auto &endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);
  EXPECT_EQ(std::any_cast<int>(endpoint.impl), 42);
  EXPECT_EQ(endpoint.sample, taggedSampleFn);
}

TEST(QuantumPlatformRuntimeEndpointTester, returnsPerQpuOverrides) {
  TestPlatform platform(2);
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 10}, /*qpuId=*/0);
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 20}, /*qpuId=*/1);

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 10);
  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(1).impl), 20);
}

TEST(QuantumPlatformRuntimeEndpointTester, fallsBackWhenOverrideMissingForQpu) {
  TestPlatform platform(2);
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 10}, /*qpuId=*/0);

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 10);

  auto &fallback = platform.getRuntimeEndpoint(1);
  auto *qpu = std::any_cast<QPU *>(fallback.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(1));
}

TEST(QuantumPlatformRuntimeEndpointTester, rejectsInvalidQpuId) {
  TestPlatform platform;
  EXPECT_THROW(platform.getRuntimeEndpoint(/*qpuId=*/1), std::invalid_argument);
  EXPECT_THROW(platform.setRuntimeEndpoint(RuntimeEndpoint{}, /*qpuId=*/1),
               std::invalid_argument);
}

// The platform owns its endpoints and hands them out by reference, so state
// written into `impl` by a launch must be visible to the next launch.
TEST(QuantumPlatformRuntimeEndpointTester,
     endpointStatePersistsAcrossLaunches) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(
      RuntimeEndpoint{.sample = countingSampleFn, .impl = 0});

  CompiledModule module;
  (void)platform.getRuntimeEndpoint().launchKernel(sample_policy{}, module, {});
  (void)platform.getRuntimeEndpoint().launchKernel(sample_policy{}, module, {});

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint().impl), 2);
}

// The lazily-created QPU fallback endpoint must be cached, too: otherwise each
// `getRuntimeEndpoint` call would hand out a fresh copy.
TEST(QuantumPlatformRuntimeEndpointTester, fallbackEndpointIsStable) {
  TestPlatform platform;
  auto *first = &platform.getRuntimeEndpoint(/*qpuId=*/0);
  auto *second = &platform.getRuntimeEndpoint(/*qpuId=*/0);
  EXPECT_EQ(first, second);
}

TEST(RuntimeEndpointWrapQpuTester, forwardsLaunchToQpu) {
  CompileTargetTestQPU qpu;
  auto endpoint = RuntimeEndpoint::wrapQPU(qpu);

  CompiledModule module;
  (void)endpoint.launchKernel(sample_policy{}, module, {});
  (void)endpoint.launchKernel(sample_policy{}, module, {});

  EXPECT_EQ(qpu.sampleLaunchCount, 2u);
}

TEST(RuntimeEndpointWrapQpuTester, forwardsLaunchThroughPlatformFallback) {
  TestPlatform platform;
  CompiledModule module;

  (void)platform.getRuntimeEndpoint(/*qpuId=*/0)
      .launchKernel(sample_policy{}, module, {});

  EXPECT_EQ(platform.getQpu(0)->sampleLaunchCount, 1u);
}

TEST(RuntimeEndpointLaunchKernelTester, throwsWhenFnUnset) {
  RuntimeEndpoint endpoint;
  CompiledModule module;
  EXPECT_THROW(endpoint.launchKernel(sample_policy{}, module, {}),
               std::runtime_error);
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesSamplePolicy) {
  testPolicyDispatch(sample_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesAsyncSamplePolicy) {
  testPolicyDispatch(async_sample_policy{.inner = {}});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesObservePolicy) {
  testPolicyDispatch(observe_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesAsyncObservePolicy) {
  testPolicyDispatch(async_observe_policy{.inner = {}});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesRunPolicy) {
  testPolicyDispatch(run_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesAsyncRunPolicy) {
  testPolicyDispatch(async_run_policy{.inner = {}});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesMsmSizePolicy) {
  testPolicyDispatch(msm_size_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesMsmPolicy) {
  testPolicyDispatch(msm_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesDemPolicy) {
  testPolicyDispatch(dem_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesPtsbeSamplePolicy) {
  testPolicyDispatch(ptsbe::sample_policy{});
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     noiseModelOpsThrowWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  expectOverrideDisabled([&] { platform.set_noise(nullptr); },
                         "Using noise models");
  expectOverrideDisabled([&] { platform.get_noise(); }, "Using noise models");
  expectOverrideDisabled([&] { platform.reset_noise(); }, "Using noise models");
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     capabilityQueriesThrowWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  expectOverrideDisabled([&] { platform.is_simulator(); }, "is_simulator");
  expectOverrideDisabled([&] { platform.is_remote(); }, "is_remote");
  expectOverrideDisabled([&] { platform.is_emulated(); }, "is_emulated");
  expectOverrideDisabled([&] { platform.get_num_qubits(); }, "get_num_qubits");
  expectOverrideDisabled([&] { platform.supports_explicit_measurements(); },
                         "supports_explicit_measurements");
  expectOverrideDisabled([&] { platform.get_remote_capabilities(); },
                         "get_remote_capabilities");
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     resourceCountLaunchThrowsWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  auto kernel = [] {};
  auto choice = [] { return false; };
  expectOverrideDisabled(
      [&] {
        (void)cudaq::detail::run_estimate_resources(kernel, platform,
                                                    "test_kernel", choice);
      },
      "Policy 'resource-count'");
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     guardSilentWhenNoEndpointOverride) {
  TestPlatform platform;

  EXPECT_NO_THROW(platform.set_noise(nullptr));
  EXPECT_TRUE(platform.is_simulator());
  EXPECT_FALSE(platform.is_remote());
  EXPECT_EQ(platform.get_num_qubits(), 30u);

  auto kernel = [] {};
  auto choice = [] { return false; };
  EXPECT_NO_THROW((void)cudaq::detail::run_estimate_resources(
      kernel, platform, "test_kernel", choice));
}

TEST(QuantumPlatformDisableEndpointOverrideTester, perQpuIsolation) {
  TestPlatform platform(2);
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/1);

  expectOverrideDisabled([&] { platform.is_remote(1); }, "is_remote");

  EXPECT_NO_THROW(platform.is_remote(0));
  EXPECT_TRUE(platform.is_simulator(0));
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     errorMessageIdentifiesOperation) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  expectOverrideDisabled([&] { platform.get_num_qubits(); }, "get_num_qubits");
  expectOverrideDisabled([&] { platform.set_noise(nullptr); },
                         "Using noise models");
}
