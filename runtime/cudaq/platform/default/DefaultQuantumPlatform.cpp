/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RuntimeTarget.h"
#include "common/Timing.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "utils/cudaq_utils.h"
#include <filesystem>
#include <fstream>

/// This file defines the default, library mode, quantum platform.
/// Its goal is to create a single QPU that is added to the quantum_platform
/// which delegates kernel execution to the current Execution Manager.

namespace {
/// The DefaultQPU models a simulated QPU by specifically
/// targeting the QIS ExecutionManager.
class DefaultQPU : public cudaq::QPU {
public:
  DefaultQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchKernel");
    return kernelFunc(args, /*isRemote=*/false);
  }

  /// Overrides setExecutionContext to forward it to the ExecutionManager
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    ScopedTraceWithContext("DefaultPlatform::setExecutionContext",
                           context->name);
    executionContext = context;
    if (noiseModel)
      executionContext->noiseModel = noiseModel;

    cudaq::getExecutionManager()->setExecutionContext(executionContext);
  }

  /// Overrides resetExecutionContext to forward to
  /// the ExecutionManager. Also handles observe post-processing
  void resetExecutionContext() override {
    ScopedTraceWithContext(
        executionContext->name == "observe" ? cudaq::TIMING_OBSERVE : 0,
        "DefaultPlatform::resetExecutionContext", executionContext->name);
    handleObservation(executionContext);
    cudaq::getExecutionManager()->resetExecutionContext();
    executionContext = nullptr;
  }
};

/// The DefaultQuantumPlatform is a quantum_platform that
/// provides a single simulated QPU, which delegates to the
/// QIS ExecutionManager.
class DefaultQuantumPlatform : public cudaq::quantum_platform {
public:
  DefaultQuantumPlatform() {
    // Populate the information and add the QPUs
    platformQPUs.emplace_back(std::make_unique<DefaultQPU>());
  }

  /// @brief Set the target backend. Here we have an opportunity
  /// to know the -qpu QPU target we are running on. This function will
  /// read in the qpu configuration file and search for the PLATFORM_QPU
  /// variable, and if found, will change from the DefaultQPU to the QPU subtype
  /// specified by that variable.
  void setTargetBackend(const std::string &backend) override {
    executionContext.set(nullptr);
    platformQPUs.clear();
    platformQPUs.emplace_back(std::make_unique<DefaultQPU>());

    CUDAQ_INFO("Backend string is {}", backend);
    std::map<std::string, std::string> configMap;
    auto mutableBackend = backend;
    if (mutableBackend.find(";") != std::string::npos) {
      auto keyVals = cudaq::split(mutableBackend, ';');
      mutableBackend = keyVals[0];
      for (std::size_t i = 1; i < keyVals.size(); i += 2)
        configMap.insert({keyVals[i], keyVals[i + 1]});
    }

    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string fileName = mutableBackend + std::string(".yml");

    /// Once we know the backend, we should search for the config file
    /// from there we can get the URL/PORT and the required MLIR pass pipeline.
    auto configFilePath = platformPath / fileName;
    CUDAQ_INFO("Config file path = {}", configFilePath.string());

    // Don't try to load something that doesn't exist.
    if (!std::filesystem::exists(configFilePath)) {
      platformQPUs.front()->setTargetBackend(backend);
      return;
    }

    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;
    runtimeTarget = std::make_unique<cudaq::RuntimeTarget>();
    runtimeTarget->config = config;
    runtimeTarget->name = mutableBackend;
    runtimeTarget->description = config.Description;
    runtimeTarget->runtimeConfig = configMap;

    if (config.BackendConfig.has_value() &&
        !config.BackendConfig->PlatformQpu.empty()) {
      auto qpuName = config.BackendConfig->PlatformQpu;
      CUDAQ_INFO("Default platform QPU subtype name: {}", qpuName);
      platformQPUs.clear();
      platformQPUs.emplace_back(cudaq::registry::get<cudaq::QPU>(qpuName));
      if (platformQPUs.front() == nullptr)
        throw std::runtime_error(
            qpuName + " is not a valid QPU name for the default platform.");
    }

    // Forward to the QPU.
    platformQPUs.front()->setTargetBackend(backend);
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(DefaultQuantumPlatform, default)
