/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/CompileTarget.h"
#include "common/CompiledModule.h"
#include "cudaq/algorithms/dem/policy.h"
#include "cudaq/algorithms/draw.h"
#include "cudaq/algorithms/msm/policy.h"
#include "cudaq/algorithms/observe/policy.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/algorithms/run/policy.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/platform/RuntimeEndpoint.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/ptsbe/policy.h"
#include <cxxabi.h>
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

  CompileTarget getCompileTarget() override {
    CompileTarget ct;
    ct.pipelineConfig.highLevelPipeline = "custom_pipeline";
    ct.fullySpecialize = false;
    ct.overrideAOTCompilation = true;
    ct.supportExplicitMeasurements = true;
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
  explicit TestPlatform(std::size_t numQpus = 1) { resetQpus(numQpus); }

  CompileTargetTestQPU *getQpu(std::size_t qpuId) {
    if (qpuId >= qpuPtrs.size() || qpuPtrs[qpuId] == nullptr)
      throw std::out_of_range("No legacy QPU at id " + std::to_string(qpuId));
    return qpuPtrs[qpuId];
  }

  void resetQpus(std::size_t numQpus = 1) {
    clearQPUs();
    qpuPtrs.clear();
    for (std::size_t i = 0; i < numQpus; ++i)
      addTestQpu();
  }

  void addTestQpu() {
    auto qpu = std::make_unique<CompileTargetTestQPU>();
    auto *raw = qpu.get();
    addQPU(std::move(qpu));
    qpuPtrs.resize(num_qpus(), nullptr);
    qpuPtrs[num_qpus() - 1] = raw;
  }

  void setEndpoint(const CompileTarget &target, RuntimeEndpoint endpoint) {
    clearQPUs();
    qpuPtrs.clear();
    addQPU(target, endpoint);
  }

  void addCustomQpu(const CompileTarget &target, RuntimeEndpoint endpoint) {
    addQPU(target, endpoint);
  }

private:
  std::vector<CompileTargetTestQPU *> qpuPtrs;
};

CompileTarget makePlatformCompileTarget() {
  CompileTarget ct;
  ct.pipelineConfig.highLevelPipeline = "custom_platform";
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
  endpoint.dispatch.set<Policy>(fn);
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

/// A test utility that expects `fn` to throw an exception of type `E` but
/// handles the case where RTTI mismatches and exception type is lost.
template <typename E, typename Fn>
std::optional<std::string> expectThrows(Fn &&fn,
                                        std::string_view exception_type) {
  try {
    fn();
  } catch (const E &e) {
    return e.what();
  } catch (...) {
    // Fallback when RTTI mismatches and exception type is lost.
    int status = -1;
    char *demangled = nullptr;

    if (auto *tinfo = abi::__cxa_current_exception_type()) {
      demangled = abi::__cxa_demangle(tinfo->name(), nullptr, nullptr, &status);
    }

    if (status != 0 || !demangled) {
      std::cerr << "\033[33m[  WARNING ]\033[0m Got exception as expected but "
                   "failed to demangle exception type. Ignoring test\n";
      return std::nullopt;
    }
    std::string type_name = demangled;
    std::free(demangled);

    // Assert that an exception was indeed thrown and its type string contains
    // 'runtime_error'
    EXPECT_NE(type_name.find(exception_type), std::string::npos)
        << "Caught unexpected exception. Expected '" << exception_type
        << "' but got '" << type_name << "'";
    return std::nullopt;
  }
  ADD_FAILURE() << "expected exception of type " << exception_type;
  return std::nullopt;
}

template <typename Fn>
void expectUnsupported(Fn &&fn, const std::string &what) {
  auto msg = expectThrows<std::runtime_error>(fn, "runtime_error");
  if (msg) {
    EXPECT_NE(msg->find(what), std::string::npos) << *msg;
    EXPECT_NE(msg->find("This QPU does not support"), std::string::npos)
        << *msg;
  }
}

} // namespace

TEST(QuantumPlatformCompileTargetTester, fallsBackToQpuWhenUnset) {
  TestPlatform platform;

  auto ct = platform.getCompileTarget();
  EXPECT_EQ(ct.pipelineConfig.highLevelPipeline, "custom_pipeline");
  EXPECT_FALSE(ct.fullySpecialize);
  EXPECT_TRUE(ct.overrideAOTCompilation);
}

TEST(QuantumPlatformCompileTargetTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  auto ct = platform.getCompileTarget();
  EXPECT_EQ(ct.pipelineConfig.highLevelPipeline, "custom_platform");
  EXPECT_TRUE(ct.fullySpecialize);
  EXPECT_FALSE(ct.overrideAOTCompilation);
  EXPECT_TRUE(ct.supportDeviceCalls);
}

TEST(QuantumPlatformCompileTargetTester,
     capabilityQueriesReportCompileTargetFlags) {
  TestPlatform platform;
  EXPECT_TRUE(platform.supports_explicit_measurements());

  platform.setEndpoint(CompileTarget{.supportExplicitMeasurements = false},
                       RuntimeEndpoint{.impl = 0});

  EXPECT_FALSE(platform.supports_explicit_measurements());
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesFallsBackToQpuWhenUnset) {
  TestPlatform platform;

  auto ct = platform.getCompileTarget();
  EXPECT_EQ(ct.pipelineConfig.highLevelPipeline, "custom_pipeline");
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesUsesPlatformOverride) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  auto ct = platform.getCompileTarget();
  EXPECT_EQ(ct.pipelineConfig.highLevelPipeline, "custom_platform");
  EXPECT_TRUE(ct.fullySpecialize);
}

TEST(QuantumPlatformCompileTargetTester, rejectsInvalidQpuId) {
  TestPlatform platform;

  expectThrows<std::invalid_argument>(
      [&] { (void)platform.getCompileTarget(/*qpu_id=*/1); },
      "invalid_argument");
}

TEST(QuantumPlatformRuntimeEndpointTester, fallsBackToQpuWhenUnset) {
  TestPlatform platform;
  auto &endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);

  auto *qpu = std::any_cast<QPU *>(endpoint.impl);
  ASSERT_NE(qpu, nullptr);
  EXPECT_EQ(qpu, platform.getQpu(0));
  EXPECT_NE(endpoint.dispatch.get<sample_policy>(), nullptr);
}

TEST(QuantumPlatformRuntimeEndpointTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  RuntimeEndpoint override;
  override.dispatch.set<sample_policy>(taggedSampleFn);
  override.impl = 42;
  platform.setEndpoint(makePlatformCompileTarget(), std::move(override));

  auto &endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);
  EXPECT_EQ(std::any_cast<int>(endpoint.impl), 42);
  EXPECT_EQ(endpoint.dispatch.get<sample_policy>(), taggedSampleFn);
}

TEST(QuantumPlatformRuntimeEndpointTester, returnsPerQpuOverrides) {
  TestPlatform platform(0);
  platform.addCustomQpu(makePlatformCompileTarget(),
                        RuntimeEndpoint{.impl = 10});
  platform.addCustomQpu(makePlatformCompileTarget(),
                        RuntimeEndpoint{.impl = 20});

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 10);
  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(1).impl), 20);
}

TEST(QuantumPlatformRuntimeEndpointTester, rejectsInvalidQpuId) {
  TestPlatform platform;
  expectThrows<std::invalid_argument>(
      [&] { platform.getRuntimeEndpoint(/*qpuId=*/1); }, "invalid_argument");
}

// The platform owns its endpoints and hands them out by reference, so state
// written into `impl` by a launch must be visible to the next launch.
TEST(QuantumPlatformRuntimeEndpointTester,
     endpointStatePersistsAcrossLaunches) {
  TestPlatform platform;
  RuntimeEndpoint counting;
  counting.dispatch.set<sample_policy>(countingSampleFn);
  counting.impl = 0;
  platform.setEndpoint(makePlatformCompileTarget(), std::move(counting));

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

// Changing the target destroys the platform's QPUs and creates new ones. The
// endpoints wrap the QPUs by reference, so replacing the QPUs must reset them
// rather than leave a wrapper pointing at a destroyed QPU.
TEST(QuantumPlatformRuntimeEndpointTester, recreatingQpusResetsEndpoints) {
  TestPlatform platform;
  RuntimeEndpoint endpoint;
  endpoint.dispatch.set<sample_policy>(taggedSampleFn);
  endpoint.impl = 42;
  platform.setEndpoint(makePlatformCompileTarget(), std::move(endpoint));
  ASSERT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint().impl), 42);

  platform.resetQpus();
  EXPECT_EQ(std::any_cast<QPU *>(platform.getRuntimeEndpoint().impl),
            platform.getQpu(0));
  EXPECT_NE(platform.getRuntimeEndpoint().dispatch.get<sample_policy>(),
            taggedSampleFn);
}

// Appending a QPU takes the next free ID, so the endpoints keyed by the
// existing IDs keep describing the same QPUs and must survive.
TEST(QuantumPlatformRuntimeEndpointTester, addingAQpuPreservesEndpoints) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(),
                       RuntimeEndpoint{.impl = 42});

  platform.addTestQpu();

  EXPECT_EQ(std::any_cast<int>(platform.getRuntimeEndpoint(0).impl), 42);
  EXPECT_EQ(std::any_cast<QPU *>(platform.getRuntimeEndpoint(1).impl),
            platform.getQpu(1));
}

// After the QPUs are replaced, launches must reach the current QPU.
TEST(QuantumPlatformRuntimeEndpointTester, launchesReachRecreatedQpu) {
  TestPlatform platform;
  CompiledModule module;

  (void)platform.getRuntimeEndpoint(/*qpuId=*/0)
      .launchKernel(sample_policy{}, module, {});
  EXPECT_EQ(platform.getQpu(0)->sampleLaunchCount, 1u);

  platform.resetQpus();

  // The replacement QPU has not been launched on yet.
  ASSERT_EQ(platform.getQpu(0)->sampleLaunchCount, 0u);
  (void)platform.getRuntimeEndpoint(/*qpuId=*/0)
      .launchKernel(sample_policy{}, module, {});
  EXPECT_EQ(platform.getQpu(0)->sampleLaunchCount, 1u);
}

TEST(RuntimeEndpointWrapQpuTester, forwardsLaunchToQpu) {
  auto qpu = std::make_unique<CompileTargetTestQPU>();
  auto endpoint = RuntimeEndpoint::fromQPU(std::move(qpu));
  auto *qpuPtr = endpoint.getQPU<CompileTargetTestQPU>();
  ASSERT_NE(qpuPtr, nullptr);

  CompiledModule module;
  (void)endpoint.launchKernel(sample_policy{}, module, {});
  (void)endpoint.launchKernel(sample_policy{}, module, {});

  EXPECT_EQ(qpuPtr->sampleLaunchCount, 2u);
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
  expectThrows<std::runtime_error>(
      [&] { endpoint.launchKernel(sample_policy{}, module, {}); },
      "runtime_error");
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

TEST(RuntimeEndpointLaunchKernelTester, dispatchesEstimatePolicy) {
  testPolicyDispatch(estimate_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesOrcaSamplePolicy) {
  testPolicyDispatch(orca::sample_policy{});
}

TEST(RuntimeEndpointLaunchKernelTester, dispatchesOrcaAsyncSamplePolicy) {
  testPolicyDispatch(orca::async_sample_policy{.inner = {}});
}

TEST(QuantumPlatformCustomEndpointTester, noiseModelOpsThrowWhenEndpointSet) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  expectUnsupported([&] { platform.set_noise(nullptr); }, "set_noise");
  EXPECT_NO_THROW(platform.reset_noise());
}

// The launch preamble queries these on every kernel run, so they must not
// throw when an endpoint override is set.
TEST(QuantumPlatformCustomEndpointTester,
     capabilityQueriesReturnDefaultsWhenEndpointSet) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  EXPECT_TRUE(platform.is_simulator());
  EXPECT_FALSE(platform.is_remote());
  EXPECT_FALSE(platform.is_emulated());
  EXPECT_TRUE(platform.supports_explicit_measurements());
  EXPECT_EQ(platform.get_noise(), nullptr);
}

// The endpoint's own flags are reported, not the (now detached) QPU's.
TEST(QuantumPlatformCustomEndpointTester,
     capabilityQueriesReportEndpointFlags) {
  TestPlatform platform;
  RuntimeEndpoint endpoint{.impl = 0};
  endpoint.isSimulator = false;
  endpoint.isRemote = true;
  endpoint.isEmulated = true;
  platform.setEndpoint(makePlatformCompileTarget(), std::move(endpoint));

  EXPECT_FALSE(platform.is_simulator());
  EXPECT_TRUE(platform.is_remote());
  EXPECT_TRUE(platform.is_emulated());
}

TEST(QuantumPlatformCustomEndpointTester, drawThrowsWhenEndpointSet) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  auto kernel = [] {};
  expectUnsupported(
      [&] { (void)cudaq::contrib::traceFromKernel(kernel, platform); },
      "configureExecutionContext");
}

TEST(QuantumPlatformCustomEndpointTester, guardSilentWhenNoEndpointOverride) {
  TestPlatform platform;

  EXPECT_NO_THROW(platform.set_noise(nullptr));
  EXPECT_TRUE(platform.is_simulator());
  EXPECT_FALSE(platform.is_remote());

  auto kernel = [] {};
  EXPECT_NO_THROW((void)cudaq::contrib::traceFromKernel(kernel, platform));
}

TEST(QuantumPlatformCustomEndpointTester, errorMessageIdentifiesOperation) {
  TestPlatform platform;
  platform.setEndpoint(makePlatformCompileTarget(), RuntimeEndpoint{.impl = 0});

  expectUnsupported([&] { platform.set_noise(nullptr); }, "set_noise");
}
