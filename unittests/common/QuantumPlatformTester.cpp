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

  void enqueue(QuantumTask &) override {}
  KernelThunkResultType unifiedLaunchModule(const AnyModule &,
                                            KernelArgs) override {
    return {};
  }
};

class TestPlatform : public quantum_platform {
public:
  explicit TestPlatform(std::size_t numQpus = 1) {
    for (std::size_t i = 0; i < numQpus; ++i)
      platformQPUs.emplace_back(std::make_unique<CompileTargetTestQPU>());
  }

  void setCompileTarget(std::optional<CompileTarget> target) {
    compileTarget = std::move(target);
  }

  void setPlatformRuntimeEndpoints(std::vector<RuntimeEndpoint> endpoints) {
    runtimeEndpoints = std::move(endpoints);
  }

  QPU *getQpu(std::size_t qpuId) { return platformQPUs[qpuId].get(); }
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
  auto endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);

  auto *qpu = std::any_cast<QPU *>(endpoint.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(0));
  EXPECT_NE(endpoint.sample, nullptr);
}

TEST(QuantumPlatformRuntimeEndpointTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  RuntimeEndpoint overrideEndpoint{.sample = taggedSampleFn, .impl = 42};
  platform.setPlatformRuntimeEndpoints({overrideEndpoint});

  auto endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);
  EXPECT_EQ(std::any_cast<int>(endpoint.impl), 42);
  EXPECT_EQ(endpoint.sample, taggedSampleFn);
}

TEST(QuantumPlatformRuntimeEndpointTester, returnsPerQpuOverrides) {
  TestPlatform platform(2);
  platform.setPlatformRuntimeEndpoints({{.impl = 10}, {.impl = 20}});

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 10);
  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(1).impl), 20);
}

TEST(QuantumPlatformRuntimeEndpointTester, fallsBackWhenOverrideMissingForQpu) {
  TestPlatform platform(2);
  platform.setPlatformRuntimeEndpoints({{.impl = 10}});

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 10);

  auto fallback = platform.getRuntimeEndpoint(1);
  auto *qpu = std::any_cast<QPU *>(fallback.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(1));
}

TEST(QuantumPlatformRuntimeEndpointTester, rejectsInvalidQpuId) {
  TestPlatform platform;
  EXPECT_THROW(platform.getRuntimeEndpoint(/*qpuId=*/1), std::invalid_argument);
}

TEST(QuantumPlatformRuntimeEndpointTester, emptyOverrideVectorFallsBackToQpu) {
  TestPlatform platform;
  platform.setPlatformRuntimeEndpoints({});

  auto endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);
  auto *qpu = std::any_cast<QPU *>(endpoint.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(0));
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
