/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "GPUEmulatedQPU.h"
#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "cuda_runtime_api.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <iostream>

using namespace cudaq;

GPUEmulatedQPU::GPUEmulatedQPU() : QPU() {}
GPUEmulatedQPU::GPUEmulatedQPU(std::size_t id) : QPU(id) {}

void GPUEmulatedQPU::enqueue(QuantumTask &task) {
  CUDAQ_INFO("Enqueue Task on QPU {}", qpu_id);
  execution_queue->enqueue(task);
}

cudaq::KernelThunkResultType
GPUEmulatedQPU::launchKernel(const cudaq::SourceModule &src,
                             cudaq::KernelArgs args) {
  CUDAQ_INFO("QPU::launchKernel GPU {}", qpu_id);
  cudaSetDevice(qpu_id);
  auto rawFn = src.getFunctionPtr();
  if (!rawFn)
    throw std::runtime_error(
        "GPUEmulatedQPU::launchKernel requires a raw kernel function "
        "pointer for kernel '" +
        src.getName() + "'.");
  auto packed = args.getPacked();
  void *argData = packed ? packed->data.data() : nullptr;
  return rawFn->getFn()(argData, /*differentMemorySpace=*/false);
}

void GPUEmulatedQPU::configureExecutionContext(
    ExecutionContext &context) const {
  CUDAQ_INFO("MultiQPUPlatform::configureExecutionContext QPU {}", qpu_id);
  if (noiseModel)
    context.noiseModel = noiseModel;

  context.executionManager = getDefaultExecutionManager();
  context.executionManager->configureExecutionContext(context);
}

void GPUEmulatedQPU::beginExecution() {
  cudaSetDevice(qpu_id);
  getExecutionContext()->executionManager->beginExecution();
}

void GPUEmulatedQPU::endExecution() {
  getExecutionContext()->executionManager->endExecution();
}

void GPUEmulatedQPU::finalizeExecutionContext(ExecutionContext &context) const {
  CUDAQ_INFO("MultiQPUPlatform::finalizeExecutionContext QPU {}", qpu_id);
  handleObservation(context);
  getExecutionContext()->executionManager->finalizeExecutionContext(context);
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::GPUEmulatedQPU, GPUEmulatedQPU)
