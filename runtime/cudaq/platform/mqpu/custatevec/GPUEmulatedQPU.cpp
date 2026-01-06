/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
protected:
  std::map<std::size_t, cudaq::ExecutionContext *> contexts;

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

  /// Overrides setExecutionContext to forward it to the ExecutionManager
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaSetDevice(qpu_id);

    CUDAQ_INFO("MultiQPUPlatform::setExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts.emplace(tid, context);
    if (noiseModel)
      contexts[tid]->noiseModel = noiseModel;

    cudaq::getExecutionManager()->setExecutionContext(contexts[tid]);
  }

  /// Overrides resetExecutionContext to forward to
  /// the ExecutionManager. Also handles observe post-processing
  void resetExecutionContext() override {
    CUDAQ_INFO("MultiQPUPlatform::resetExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto ctx = contexts[tid];
    handleObservation(ctx);
    cudaq::getExecutionManager()->resetExecutionContext();
    contexts[tid] = nullptr;
    contexts.erase(tid);
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, GPUEmulatedQPU, GPUEmulatedQPU)
