/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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
#include "cudaq/spin_op.h"
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
  GPUEmulatedQPU() = default;
  GPUEmulatedQPU(std::size_t id) : QPU(id) {}

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("Enqueue Task on QPU {}", qpu_id);
    cudaSetDevice(qpu_id);
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t, std::uint64_t) override {
    cudaq::info("QPU::launchKernel GPU {}", qpu_id);
    cudaSetDevice(qpu_id);
    kernelFunc(args);
  }

  /// Overrides setExecutionContext to forward it to the ExecutionManager
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaSetDevice(qpu_id);

    cudaq::info("MultiQPUPlatform::setExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts.emplace(tid, context);
    if (noiseModel)
      contexts[tid]->noiseModel = noiseModel;

    cudaq::getExecutionManager()->setExecutionContext(contexts[tid]);
  }

  /// Overrides resetExecutionContext to forward to
  /// the ExecutionManager. Also handles observe post-processing
  void resetExecutionContext() override {
    cudaq::info("MultiQPUPlatform::resetExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    auto ctx = contexts[tid];
    handleObservation(ctx);
    cudaq::getExecutionManager()->resetExecutionContext();
    contexts[tid] = nullptr;
    contexts.erase(tid);
  }
};

class MultiQPUQuantumPlatform : public cudaq::quantum_platform {
public:
  ~MultiQPUQuantumPlatform() = default;
  MultiQPUQuantumPlatform() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    auto envVal = spdlog::details::os::getenv("CUDAQ_MQPU_NGPUS");
    if (!envVal.empty()) {
      int specifiedNDevices = 0;
      try {
        specifiedNDevices = std::stoi(envVal);
      } catch (...) {
        throw std::runtime_error(
            "Invalid CUDAQ_MQPU_NGPUS environment variable, must be integer.");
      }

      if (specifiedNDevices < nDevices)
        nDevices = specifiedNDevices;
    }

    if (nDevices == 0)
      throw std::runtime_error("No GPUs available to instantiate platform.");

    // Add a QPU for each GPU.
    for (int i = 0; i < nDevices; i++)
      platformQPUs.emplace_back(std::make_unique<GPUEmulatedQPU>(i));

    platformNumQPUs = platformQPUs.size();
    platformCurrentQPU = 0;
  }

  bool supports_task_distribution() const override { return true; }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
