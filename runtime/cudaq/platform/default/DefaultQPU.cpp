/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"
#include "common/ExecutionContext.h"
#include "common/Timing.h"
#include "cudaq/runtime/logger/logger.h"

cudaq::DefaultQPU::~DefaultQPU() = default;

void cudaq::DefaultQPU::enqueue(QuantumTask &task) {
  execution_queue->enqueue(task);
}

cudaq::KernelThunkResultType
cudaq::DefaultQPU::unifiedLaunchModule(const cudaq::AnyModule &module,
                                       cudaq::KernelArgs args) {
  if (!std::holds_alternative<cudaq::SourceModule>(module))
    return runJITCompiledModule(std::get<cudaq::CompiledModule>(module), args);

  const auto &src = std::get<cudaq::SourceModule>(module);
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::unifiedLaunchModule");
  auto rawFn = src.getFunctionPtr();
  if (!rawFn)
    throw std::runtime_error(
        "DefaultQPU::unifiedLaunchModule requires a raw kernel function "
        "pointer for kernel '" +
        src.getName() + "'.");
  auto packed = args.getPacked();
  void *argData = packed ? packed->data.data() : nullptr;
  return rawFn->getFn()(argData, /*isRemote=*/false);
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
  handleObservation(context);

  getExecutionContext()->executionManager->finalizeExecutionContext(context);
}
