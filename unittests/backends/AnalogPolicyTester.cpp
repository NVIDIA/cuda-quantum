/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogHamiltonian.h"
#include "common/AnalogRemoteRESTQPU.h"
#include <gtest/gtest.h>

namespace {

class LocalAhsQpu : public cudaq::AnalogRemoteRESTQPU {
public:
  LocalAhsQpu() { emulate = true; }
};

// The launch path itself is covered by `targettests/analog`, which exercises it
// from an `nvq++`-built binary with a complete runtime.
TEST(AnalogPolicyTester, CompileTargetPreservesSourceModule) {
  cudaq::AnalogRemoteRESTQPU qpu;
  auto target = qpu.getCompileTarget();
  EXPECT_FALSE(target.overrideAOTCompilation);
}

TEST(AnalogPolicyTester, MissingEmulatorReportsRequirement) {
  LocalAhsQpu qpu;
  EXPECT_TRUE(qpu.isEmulated());
  EXPECT_FALSE(qpu.isRemote());
  std::string payload = cudaq::ahs::toJsonString(cudaq::ahs::Program{});
  cudaq::KernelArgs args(
      cudaq::KernelArgs::PackedArgs{payload.data(), payload.size(), 0});
  cudaq::CompiledModule module(
      cudaq::SourceModule("__analog_hamiltonian_kernel__no_dynamics"));
  try {
    qpu.launchKernel(cudaq::sample_policy{}, module, args);
    FAIL() << "An AHS backend without an emulator must reject emulation.";
  } catch (const std::runtime_error &error) {
    EXPECT_STREQ(error.what(),
                 "Local emulation is not available for this target. It "
                 "requires a target with emulation support and the CUDA-Q "
                 "dynamics backend library.");
  }
}

TEST(AnalogPolicyTester, EmulatedAsyncLaunchIsRejected) {
  // Emulated targets run asynchronous work on the platform queue instead.
  LocalAhsQpu qpu;
  std::string payload = cudaq::ahs::toJsonString(cudaq::ahs::Program{});
  cudaq::KernelArgs args(
      cudaq::KernelArgs::PackedArgs{payload.data(), payload.size(), 0});
  cudaq::CompiledModule module(
      cudaq::SourceModule("__analog_hamiltonian_kernel__async"));
  EXPECT_THROW(qpu.launchKernel(cudaq::async_sample_policy{}, module, args),
               std::runtime_error);
}

TEST(AnalogPolicyTester, UnknownEmulationEngineIsUnavailable) {
  EXPECT_EQ(cudaq::analog::loadEngine("no-such-engine"), nullptr);
}

} // namespace
