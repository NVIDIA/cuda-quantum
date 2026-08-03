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
#include <cxxabi.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <string>

using namespace cudaq;

namespace {

class CompileTargetTestQPU : public QPU {
public:
  /// Number of times `launchKernel(sample_policy)` was called on this QPU.
  std::size_t sampleLaunchCount = 0;

  CompileTarget getCompileTarget(const sample_policy &) override {
    CompileTarget ct;
    ct.emitJit = true;
    ct.fullySpecialize = false;
    ct.overrideAOTCompilation = true;
    return ct;
  }

  CompileTarget getCompileTarget(const other_policies &,
                                 ExecutionContext *) override {
    CompileTarget ct;
    ct.emitResourceCounts = true;
    ct.emitJit = false;
    ct.fullySpecialize = false;
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

  explicit TestPlatform(std::size_t numQpus = 1) { resetQpus(numQpus); }

  CompileTargetTestQPU *getQpu(std::size_t qpuId) {
    return static_cast<CompileTargetTestQPU *>(&getQPU(qpuId));
  }

  void resetQpus(std::size_t numQpus = 1) {
    clearQPUs();
    for (std::size_t i = 0; i < numQpus; ++i)
      addTestQpu();
  }

  void addTestQpu() { addQPU(std::make_unique<CompileTargetTestQPU>()); }
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
void expectOverrideDisabled(Fn &&fn, const std::string &what) {
  auto msg = expectThrows<std::runtime_error>(fn, "runtime_error");
  if (msg) {
    EXPECT_NE(msg->find(what), std::string::npos) << *msg;
    EXPECT_NE(msg->find("manually setting a runtime endpoint"),
              std::string::npos)
        << *msg;
  }
}

} // namespace

TEST(QuantumPlatformCompileTargetTester, fallsBackToQpuWhenUnset) {
  TestPlatform platform;
  sample_policy policy{.kernelName = "test_kernel"};

  auto ct = platform.getCompileTarget(policy);
  EXPECT_TRUE(ct.emitJit);
  EXPECT_FALSE(ct.fullySpecialize);
  EXPECT_TRUE(ct.overrideAOTCompilation);
}

TEST(QuantumPlatformCompileTargetTester, usesPlatformOverrideWhenSet) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  sample_policy policy{.kernelName = "test_kernel"};

  auto ct = platform.getCompileTarget(policy);
  EXPECT_FALSE(ct.emitJit);
  EXPECT_TRUE(ct.fullySpecialize);
  EXPECT_FALSE(ct.overrideAOTCompilation);
  EXPECT_TRUE(ct.supportDeviceCalls);
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesFallsBackToQpuWhenUnset) {
  TestPlatform platform;
  other_policies policy;

  auto ct = platform.getCompileTarget(policy);
  EXPECT_TRUE(ct.emitResourceCounts);
  EXPECT_FALSE(ct.emitJit);
}

TEST(QuantumPlatformCompileTargetTester, otherPoliciesUsesPlatformOverride) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  other_policies policy;

  auto ct = platform.getCompileTarget(policy);
  EXPECT_FALSE(ct.emitJit);
  EXPECT_TRUE(ct.fullySpecialize);
  EXPECT_FALSE(ct.emitResourceCounts);
}

TEST(QuantumPlatformCompileTargetTester, rejectsInvalidQpuId) {
  TestPlatform platform;
  sample_policy policy{.kernelName = "test_kernel"};

  expectThrows<std::invalid_argument>(
      [&] { (void)platform.getCompileTarget(policy, /*qpu_id=*/1); },
      "invalid_argument");
}

TEST(QuantumPlatformCompileTargetTester, clearingOverrideFallsBackToQpu) {
  TestPlatform platform;
  platform.setCompileTarget(makePlatformCompileTarget());
  sample_policy policy{.kernelName = "test_kernel"};

  platform.setCompileTarget(std::nullopt);

  auto ct = platform.getCompileTarget(policy);
  EXPECT_TRUE(ct.emitJit);
  EXPECT_TRUE(ct.overrideAOTCompilation);
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
  platform.setRuntimeEndpoint(std::move(override));

  auto &endpoint = platform.getRuntimeEndpoint(/*qpuId=*/0);
  EXPECT_EQ(std::any_cast<int>(endpoint.impl), 42);
  EXPECT_EQ(endpoint.dispatch.get<sample_policy>(), taggedSampleFn);
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
  expectThrows<std::invalid_argument>(
      [&] { platform.getRuntimeEndpoint(/*qpuId=*/1); }, "invalid_argument");
  expectThrows<std::invalid_argument>(
      [&] { platform.setRuntimeEndpoint(RuntimeEndpoint{}, /*qpuId=*/1); },
      "invalid_argument");
}

// The platform owns its endpoints and hands them out by reference, so state
// written into `impl` by a launch must be visible to the next launch.
TEST(QuantumPlatformRuntimeEndpointTester,
     endpointStatePersistsAcrossLaunches) {
  TestPlatform platform;
  RuntimeEndpoint counting;
  counting.dispatch.set<sample_policy>(countingSampleFn);
  counting.impl = 0;
  platform.setRuntimeEndpoint(std::move(counting));

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
//
// A manually set endpoint makes the reset directly observable: its `impl`
// holds an `int`, so if it survived the reset the `any_cast<QPU *>` below
// would throw instead of yielding the newly created QPU.
TEST(QuantumPlatformRuntimeEndpointTester, recreatingQpusResetsEndpoints) {
  TestPlatform platform;
  RuntimeEndpoint endpoint;
  endpoint.dispatch.set<sample_policy>(taggedSampleFn);
  endpoint.impl = 42;
  platform.setRuntimeEndpoint(std::move(endpoint));
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
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 42});

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

TEST(QuantumPlatformDisableEndpointOverrideTester,
     noiseModelOpsThrowWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  expectOverrideDisabled([&] { platform.set_noise(nullptr); },
                         "Using noise models");
  expectOverrideDisabled([&] { platform.reset_noise(); }, "Using noise models");
}

TEST(QuantumPlatformDisableEndpointOverrideTester,
     capabilityQueriesThrowWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  expectOverrideDisabled([&] { platform.is_simulator(); }, "is_simulator");
  expectOverrideDisabled([&] { platform.get_num_qubits(); }, "get_num_qubits");
  expectOverrideDisabled([&] { platform.get_remote_capabilities(); },
                         "get_remote_capabilities");
}

// The launch preamble queries these on every kernel run, so they must not
// throw when an endpoint override is set. Instead they warn and report safe
// defaults, since there is no backing QPU to forward to.
TEST(QuantumPlatformDisableEndpointOverrideTester,
     capabilityQueriesReturnDefaultsWhenEndpointSet) {
  TestPlatform platform;
  platform.setRuntimeEndpoint(RuntimeEndpoint{.impl = 0}, /*qpuId=*/0);

  EXPECT_FALSE(platform.is_remote());
  EXPECT_FALSE(platform.is_emulated());
  EXPECT_FALSE(platform.supports_explicit_measurements());
  EXPECT_EQ(platform.get_noise(), nullptr);
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

  expectOverrideDisabled([&] { platform.is_simulator(1); }, "is_simulator");

  EXPECT_NO_THROW(platform.is_simulator(0));
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
