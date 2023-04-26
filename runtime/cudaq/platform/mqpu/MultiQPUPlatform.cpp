/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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
// We want to kick off CUDA lazy initialization,
// flip this to true once we do
static bool devicesWarmedUp = false;

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
    if (ctx && ctx->name == "observe") {
      double sum = 0.0;
      if (!ctx->spin.has_value())
        throw std::runtime_error(
            "Observe ExecutionContext specified without a cudaq::spin_op.");

      std::vector<cudaq::ExecutionResult> results;
      cudaq::spin_op &H = *ctx->spin.value();

      // If the backend supports the observe task,
      // let it compute the expectation value instead of
      // manually looping over terms, applying basis change ops,
      // and computing <ZZ..ZZZ>
      if (ctx->canHandleObserve) {
        auto [exp, data] = cudaq::measure(H);
        results.emplace_back(data.to_map(), H.to_string(false), exp);
        ctx->expectationValue = exp;
        ctx->result = cudaq::sample_result(results);
      } else {
        H.for_each_term([&](cudaq::spin_op &term) {
          if (term.is_identity())
            sum += term.get_coefficient().real();
          else {

            auto [exp, data] = cudaq::measure(term);
            results.emplace_back(data.to_map(), term.to_string(), exp);
            sum += term.get_coefficient().real() * exp;
          }
        });

        ctx->expectationValue = sum;
        ctx->result = cudaq::sample_result(sum, results);
      }
    }

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

    if (!devicesWarmedUp) {
      // Warm up the GPUs so we don't have any lazy init issues.
      std::vector<std::future<void>> futures;
      for (int i = 0; i < nDevices; i++) {
        futures.emplace_back(std::async(std::launch::async, [i]() {
          auto warmUpSim = cudaq::getExecutionManager();

          cudaSetDevice(i);

          // Warm up the GPUs via an allocation / deallocation.
          cudaq::info("Warm up Emulated QPU (GPU) {}.", i);
          std::array<std::size_t, 1> qbits{warmUpSim->getAvailableIndex()};
          warmUpSim->returnQudit({2, qbits[0]});
        }));
      }

      // Sync up the threads
      for (auto &f : futures)
        f.get();

      cudaSetDevice(0);
      devicesWarmedUp = true;
    }

    // Add a QPU for each GPU.
    for (int i = 0; i < nDevices; i++)
      platformQPUs.emplace_back(std::make_unique<GPUEmulatedQPU>(i));

    platformNumQPUs = platformQPUs.size();
    platformCurrentQPU = 0;
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
