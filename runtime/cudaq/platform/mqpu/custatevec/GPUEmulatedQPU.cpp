/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/NoiseModel.h"
#include "cuda_runtime_api.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <iostream>
#include <spdlog/cfg/env.h>

namespace {

/// @brief This QPU implementation enqueues kernel
/// execution tasks and sets the CUDA GPU device that it
/// represents. There is a GPUEmulatedQPU per available GPU.
class GPUEmulatedQPU : public cudaq::QPU {
public:
  GPUEmulatedQPU() : QPU(){};
  GPUEmulatedQPU(std::size_t id) : QPU(id) {}

  void enqueue(cudaq::QuantumTask &task) override {
    // Note: enqueue is executed on the main thread, not the QPU execution
    // thread. Hence, do not set the CUDA device here.
    CUDAQ_INFO("Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t, std::uint64_t,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("QPU::launchKernel GPU {}", qpu_id);
    cudaSetDevice(qpu_id);
    return kernelFunc(args, /*differentMemorySpace=*/false);
  }

  void configureExecutionContext(cudaq::ExecutionContext &context) override {
    cudaSetDevice(qpu_id);

    CUDAQ_INFO("MultiQPUPlatform::prepareExecutionContext QPU {}", qpu_id);
    if (noiseModel)
      context.noiseModel = noiseModel;

    // TODO: remove execution context from ExecutionManager
    cudaq::getExecutionManager()->setExecutionContext(&context);
  }

  void processExecutionResults(cudaq::ExecutionContext &context) override {
    CUDAQ_INFO("MultiQPUPlatform::processExecutionResults QPU {}", qpu_id);

    handleObservation(context);
    cudaq::getExecutionManager()->resetExecutionContext();
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, GPUEmulatedQPU, GPUEmulatedQPU)
