/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"

void cudaq::details::DefaultQPU::enqueue(cudaq::QuantumTask &task) {
  execution_queue->enqueue(task);
}

cudaq::KernelThunkResultType cudaq::details::DefaultQPU::launchKernel(
    const std::string &name, cudaq::KernelThunkType kernelFunc, void *args,
    std::uint64_t argsSize, std::uint64_t resultOffset,
    const std::vector<void *> &rawArgs) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchKernel");
  return kernelFunc(args, /*isRemote=*/false);
}

void cudaq::details::DefaultQPU::configureExecutionContext(
    cudaq::ExecutionContext &context) const {
  ScopedTraceWithContext("DefaultPlatform::prepareExecutionContext",
                         context.name);
  if (noiseModel)
    context.noiseModel = noiseModel;

  context.executionManager = cudaq::getDefaultExecutionManager();
  context.executionManager->configureExecutionContext(context);
}

void cudaq::details::DefaultQPU::beginExecution() {
  cudaq::getExecutionContext()->executionManager->beginExecution();
}

void cudaq::details::DefaultQPU::endExecution() {
  cudaq::getExecutionContext()->executionManager->endExecution();
}

void cudaq::details::DefaultQPU::finalizeExecutionContext(
    cudaq::ExecutionContext &context) const {
  ScopedTraceWithContext(context.name == "observe" ? cudaq::TIMING_OBSERVE : 0,
                         "DefaultPlatform::finalizeExecutionContext",
                         context.name);
  handleObservation(context);

  cudaq::getExecutionContext()->executionManager->finalizeExecutionContext(
      context);
}
