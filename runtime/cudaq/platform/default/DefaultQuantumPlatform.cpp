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
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include <fstream>

/// This file defines the default, library mode, quantum platform.
/// Its goal is to create a single QPU that is added to the quantum_platform
/// which delegates kernel execution to the current Execution Manager.

LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

namespace {
/// The DefaultQPU models a simulated QPU by specifically
/// targeting the QIS ExecutionManager.
class DefaultQPU : public cudaq::QPU {
public:
  DefaultQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t, std::uint64_t) override {
    cudaq::ScopedTrace trace("QPU::launchKernel");
    kernelFunc(args);
  }

  /// Overrides setExecutionContext to forward it to the ExecutionManager
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::ScopedTrace trace("DefaultPlatform::setExecutionContext",
                             context->name);
    executionContext = context;
    if (noiseModel)
      executionContext->noiseModel = noiseModel;

    cudaq::getExecutionManager()->setExecutionContext(executionContext);
  }

  /// Overrides resetExecutionContext to forward to
  /// the ExecutionManager. Also handles observe post-processing
  void resetExecutionContext() override {
    cudaq::ScopedTrace trace("DefaultPlatform::resetExecutionContext",
                             executionContext->name);
    handleObservation(executionContext);
    cudaq::getExecutionManager()->resetExecutionContext();
    executionContext = nullptr;
  }
};

constexpr char platformQPU[] = "PLATFORM_QPU";

/// The DefaultQuantumPlatform is a quantum_platform that
/// provides a single simulated QPU, which delegates to the
/// QIS ExecutionManager.
class DefaultQuantumPlatform : public cudaq::quantum_platform {
public:
  DefaultQuantumPlatform() {
    // Populate the information and add the QPUs
    platformQPUs.emplace_back(std::make_unique<DefaultQPU>());
    platformNumQPUs = platformQPUs.size();
  }

  /// @brief Set the target backend. Here we have an opportunity
  /// to know the -qpu QPU target we are running on. This function will
  /// read in the qpu configuration file and search for the PLATFORM_QPU
  /// variable, and if found, will change from the DefaultQPU to the QPU subtype
  /// specified by that variable.
  void setTargetBackend(const std::string &backend) override {
    platformQPUs.clear();
    platformQPUs.emplace_back(std::make_unique<DefaultQPU>());

    cudaq::info("Backend string is {}", backend);
    auto mutableBackend = backend;
    if (mutableBackend.find(";") != std::string::npos) {
      mutableBackend = cudaq::split(mutableBackend, ';')[0];
    }

    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string fileName = mutableBackend + std::string(".config");

    /// Once we know the backend, we should search for the config file
    /// from there we can get the URL/PORT and the required MLIR pass pipeline.
    auto configFilePath = platformPath / fileName;
    cudaq::info("Config file path = {}", configFilePath.string());

    // Don't try to load something that doesn't exist.
    if (!std::filesystem::exists(configFilePath)) {
      platformQPUs.front()->setTargetBackend(backend);
      return;
    }

    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());

    auto lines = cudaq::split(configContents, '\n');
    for (auto &line : lines) {
      if (line.find(platformQPU) != std::string::npos) {
        auto keyVal = cudaq::split(line, '=');
        auto qpuName = keyVal[1];
        cudaq::info("Default platform QPU subtype name: {}", qpuName);
        platformQPUs.clear();
        platformQPUs.emplace_back(cudaq::registry::get<cudaq::QPU>(qpuName));
        if (platformQPUs.front() == nullptr)
          throw std::runtime_error(
              qpuName + " is not a valid QPU name for the default platform.");
      }
    }

    // Forward to the QPU.
    platformQPUs.front()->setTargetBackend(backend);
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(DefaultQuantumPlatform, default)
