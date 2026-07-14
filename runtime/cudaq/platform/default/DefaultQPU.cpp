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

cudaq::DefaultQPU::~DefaultQPU() = default;

void cudaq::DefaultQPU::enqueue(QuantumTask &task) {
  execution_queue->enqueue(task);
}

cudaq::KernelThunkResultType
cudaq::DefaultQPU::unifiedLaunchModule(const cudaq::AnyModule &module,
                                       cudaq::KernelArgs args) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::unifiedLaunchModule");

  std::optional<FatQuakeModule::FunctionPtrArtifact> rawFn;
  if (std::holds_alternative<SourceModule>(module)) {
    rawFn = std::get<SourceModule>(module).getFunctionPtr();
    assert(rawFn && "SourceModule must have a valid AOT-compiled thunk");
  } else {
    auto &compiled = std::get<CompiledModule>(module);
    rawFn = compiled.getFunctionPtr();
    if (!rawFn)
      return runJITCompiledModule(compiled, args);
  }

  auto packed = args.getPacked();
  void *argData = packed ? packed->data.data() : nullptr;
  // Mark the kernel frame so the simulator defers (rather than throws)
  // exceptions while the JIT'd kernel runs; rethrowDeferredKernelException()
  // below surfaces any such error from this C++ frame.
  InKernelLaunchScope kernelFrame;
  auto result = rawFn->getFn()(argData, /*isRemote=*/false);
  // Surface any error the kernel deferred rather than threw (see
  // QPU::rethrowDeferredKernelException).
  rethrowDeferredKernelException();
  return result;
}

cudaq::sample_result
cudaq::DefaultQPU::launchKernel(const cudaq::sample_policy &policy,
                                const cudaq::CompiledModule &module,
                                cudaq::KernelArgs args) {
  CUDAQ_INFO("DefaultQPU::launchKernel {}", policy.name);
  return cudaq::ExecutionManager::with_default_em(
      policy,
      [this, &module, &args]() { this->unifiedLaunchModule(module, args); });
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
  return cudaq::ExecutionManager::with_default_em(
      policy,
      [this, &module, &args]() { this->unifiedLaunchModule(module, args); });
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
  return cudaq::ExecutionManager::with_default_em(
      policy,
      [this, &module, &args]() { this->unifiedLaunchModule(module, args); });
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::DefaultQPU::getCompileTarget(const sample_policy &policy) {
  return getDefaultCompileTarget(policy);
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::DefaultQPU::getCompileTarget(const observe_policy &policy) {
  return getDefaultCompileTarget(policy);
}

std::unique_ptr<cudaq::CompileTarget>
cudaq::DefaultQPU::getCompileTarget(const dem_policy &policy) {
  return getDefaultCompileTarget(policy);
}

std::unique_ptr<cudaq::CompileTarget>
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
