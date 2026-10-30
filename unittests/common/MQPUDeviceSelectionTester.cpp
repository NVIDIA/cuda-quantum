/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#undef CUDAQ_LIBRARY_MODE
#include "DefaultQPU.h"
#include "cudaq/algorithms/get_state.h"
#include "cudaq/algorithms/observe.h"
#include "cudaq/algorithms/sample.h"
#include "cudaq/platform/mqpu/helpers/MQPUUtils.h"
#include <gtest/gtest.h>

namespace cudaq::detail {
class PlatformTestAccess {
public:
  static void setBackend(quantum_platform &platform, const std::string &name) {
    platform.setTargetBackend(name);
  }
};
} // namespace cudaq::detail

namespace {
constexpr int numDevices = 4;
thread_local int selectedDevice = 0;

class TestOrcaQPU : public cudaq::DefaultQPU {
public:
  bool isRemote() override { return true; }
};

static cudaq::QPU::RegistryType::Add<TestOrcaQPU> registerOrca("orca");

void kernelBody() {
  EXPECT_EQ(selectedDevice, cudaq::getCurrentQpuId());
  auto *em = cudaq::getExecutionManager();
  cudaq::QuditInfo qubit{2, em->allocateQudit()};
  em->apply("x", {}, {}, {qubit});
  em->returnQudit(qubit);
}

cudaq::KernelThunkResultType kernelThunk(void *, bool) {
  kernelBody();
  return {};
}

// Mimic the compiler-generated launch stub so no JIT is needed.
void kernel() {
  auto *ctx = cudaq::getExecutionContext();
  if (ctx->executeKernelApi) {
    cudaq::CompiledModule module{
        cudaq::SourceModule{"device_selection", kernelThunk}};
    ctx->executeKernelApi(module, {});
  } else {
    kernelBody();
  }
}

class MQPUDeviceSelection : public testing::TestWithParam<bool> {
protected:
  std::unique_ptr<cudaq::detail::with_platform_in_library_mode> libraryMode;

  void SetUp() override {
    auto &platform = cudaq::get_platform();
    ASSERT_EQ(platform.num_qpus(), numDevices);
    if (GetParam())
      libraryMode =
          std::make_unique<cudaq::detail::with_platform_in_library_mode>(
              platform);
    // Reset each worker's device so every test starts from a cold binding.
    for (std::size_t qpu = 0; qpu < platform.num_qpus(); ++qpu) {
      std::promise<void> reset;
      auto done = reset.get_future();
      cudaq::QuantumTask task = [&reset] {
        selectedDevice = 0;
        reset.set_value();
      };
      platform.enqueueAsyncTask(qpu, task);
      done.get();
    }
  }
};
} // namespace

// Host-only CUDA helpers; the binding is thread-local, as in CUDA.
int cudaq::getCudaDeviceCount() { return numDevices; }
void cudaq::setCudaDevice(int device) { selectedDevice = device; }

TEST_P(MQPUDeviceSelection, ObserveAsyncBindsEachWorkerWithoutWarmup) {
  std::vector<cudaq::async_observe_result> results;
  for (std::size_t qpu = 0; qpu < numDevices; ++qpu)
    results.emplace_back(cudaq::observe_async(qpu, kernel, cudaq::spin::z(0)));
  for (auto &result : results)
    EXPECT_NEAR(result.get().expectation(), -1.0, 1e-12);
}

TEST_P(MQPUDeviceSelection, SampleAsyncBindsEachWorkerWithoutWarmup) {
  std::vector<cudaq::async_sample_result> results;
  for (std::size_t qpu = 0; qpu < numDevices; ++qpu)
    results.emplace_back(cudaq::sample_async(10, qpu, kernel));
  for (auto &result : results)
    EXPECT_EQ(result.get().count("1"), 10);
}

TEST_P(MQPUDeviceSelection, GetStateAsyncStillBindsEachWorker) {
  std::vector<cudaq::async_state_result> results;
  for (std::size_t qpu = 0; qpu < numDevices; ++qpu)
    results.emplace_back(cudaq::get_state_async(qpu, kernel));
  for (auto &result : results)
    EXPECT_NEAR(std::abs(result.get()[1]), 1.0, 1e-12);
}

TEST_P(MQPUDeviceSelection, TargetSwitchOnlyBindsGpuQpus) {
  auto &platform = cudaq::get_platform();
  cudaq::detail::PlatformTestAccess::setBackend(platform,
                                                "orca;url;localhost:1234");
  selectedDevice = -1;
  platform.prepareExecution(0);
  EXPECT_EQ(selectedDevice, -1);

  // Switching back to GPU QPUs must re-enable device binding.
  cudaq::detail::PlatformTestAccess::setBackend(platform, "qpp-cpu");
  for (std::size_t qpu = 0; qpu < numDevices; ++qpu)
    EXPECT_NEAR(cudaq::observe_async(qpu, kernel, cudaq::spin::z(0))
                    .get()
                    .expectation(),
                -1.0, 1e-12);
}

INSTANTIATE_TEST_SUITE_P(LaunchPaths, MQPUDeviceSelection,
                         testing::Values(false, true), [](const auto &info) {
                           return info.param ? "Library" : "Compiled";
                         });
