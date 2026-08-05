/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"
#include "common/CompiledModule.h"
#include "common/ExecutionContext.h"
#include "common/Timing.h"
#include "cudaq/algorithms/policies.h"
#include "cudaq/platform.h"
#include "cudaq/runtime/logger/logger.h"

namespace nvqir {
void setRandomSeed(std::size_t seed);
}

cudaq::DefaultQPU::~DefaultQPU() = default;

void cudaq::DefaultQPU::enqueue(QuantumTask &task) {
  execution_queue->enqueue(task);
}

void cudaq::DefaultQPU::onRandomSeedSet(std::size_t seed) {
  // QPP's random generator is thread-local. Seed it on the QPU execution
  // thread as well, which is where asynchronous algorithm tasks run.
  if (std::this_thread::get_id() == getExecutionThreadId()) {
    nvqir::setRandomSeed(seed);
    return;
  }

  std::promise<void> seeded;
  auto completed = seeded.get_future();
  QuantumTask task = [seed, &seeded]() {
    try {
      nvqir::setRandomSeed(seed);
      seeded.set_value();
    } catch (...) {
      seeded.set_exception(std::current_exception());
    }
  };
  enqueue(task);
  completed.get();
}

cudaq::KernelThunkResultType
cudaq::DefaultQPU::unifiedLaunchModule(const cudaq::AnyModule &module,
                                       cudaq::KernelArgs args) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::unifiedLaunchModule");

  if (std::holds_alternative<SourceModule>(module)) {
    auto rawFn = std::get<SourceModule>(module).getFunctionPtr();
    assert(rawFn && "SourceModule must have a valid AOT-compiled thunk");
    return executeFunctionPtrBinary(*rawFn, args);
  }

  auto &compiled = std::get<CompiledModule>(module);
  return executeCompiledModule(compiled, args);
}

cudaq::sample_result
cudaq::DefaultQPU::launchKernel(const cudaq::sample_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::async_sample_result
cudaq::DefaultQPU::launchKernel(const async_sample_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  throw std::runtime_error(
      "DefaultQPU does not support launching the async_sample_policy.");
}

cudaq::observe_result
cudaq::DefaultQPU::launchKernel(const cudaq::observe_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::run_result
cudaq::DefaultQPU::launchKernel(const cudaq::run_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::async_run_policy::result_type
cudaq::DefaultQPU::launchKernel(const async_run_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  throw std::runtime_error(
      "DefaultQPU does not support launching the async_run_policy.");
}

cudaq::msm_dimensions
cudaq::DefaultQPU::launchKernel(const cudaq::msm_size_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::msm_result
cudaq::DefaultQPU::launchKernel(const cudaq::msm_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::async_observe_result
cudaq::DefaultQPU::launchKernel(const async_observe_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  throw std::runtime_error(
      "DefaultQPU does not support launching the async_observe_policy.");
}

cudaq::dem_result
cudaq::DefaultQPU::launchKernel(const cudaq::dem_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::ptsbe::sample_policy::result_type
cudaq::DefaultQPU::launchKernel(const cudaq::ptsbe::sample_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(policy, [&module, &args]() {
    [[maybe_unused]] auto res = executeCompiledModule(module, args);
  });
}

cudaq::CompileTarget
cudaq::DefaultQPU::getCompileTarget(const sample_policy &policy) {
  return getDefaultCompileTarget(policy);
}

cudaq::CompileTarget
cudaq::DefaultQPU::getCompileTarget(const observe_policy &policy) {
  return getDefaultCompileTarget(policy);
}

cudaq::CompileTarget
cudaq::DefaultQPU::getCompileTarget(const run_policy &policy) {
  return getDefaultCompileTarget(policy);
}

cudaq::CompileTarget
cudaq::DefaultQPU::getCompileTarget(const dem_policy &policy) {
  return getDefaultCompileTarget(policy);
}

cudaq::CompileTarget
cudaq::DefaultQPU::getCompileTarget(const other_policies &policy,
                                    ExecutionContext *context) {
  return getDefaultCompileTarget(policy, context);
}

void cudaq::DefaultQPU::configureExecutionContext(
    ExecutionContext &context) const {
  ScopedTraceWithContext("DefaultPlatform::prepareExecutionContext",
                         context.name);
  if (noiseModel)
    context.noiseModel = noiseModel;

  context.executionManager = getDefaultExecutionManager();
  context.executionManager->configureExecutionContext(context);
}

void cudaq::DefaultQPU::beginExecution() {
  getExecutionContext()->executionManager->beginExecution();
}

void cudaq::DefaultQPU::endExecution() {
  getExecutionContext()->executionManager->endExecution();
}

void cudaq::DefaultQPU::finalizeExecutionContext(
    ExecutionContext &context) const {
  ScopedTraceWithContext(context.name == "observe" ? TIMING_OBSERVE : 0,
                         "DefaultPlatform::finalizeExecutionContext",
                         context.name);
  getExecutionContext()->executionManager->finalizeExecutionContext(context);
}
