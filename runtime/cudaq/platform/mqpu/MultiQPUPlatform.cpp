/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DefaultQPU.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/RuntimeTarget.h"
#include "helpers/MQPUUtils.h"
#include "cudaq/Target/TargetConfigYaml.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/simulators.h"
#include <filesystem>

// Note: LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType) is intentionally
// NOT placed here. The canonical QPU registry instance lives in
// quantum_platform.cpp (libcudaq). With LLVM 22's static-inline Head/Tail
// pointers in llvm::Registry, having the instantiation in multiple DSOs can
// cause registry fragmentation — nodes added via cudaq_add_qpu_node (which
// targets libcudaq's registry) would be invisible to code in this DSO if the
// linker kept separate copies. A single instantiation in libcudaq avoids this.

namespace {
class MultiQPUQuantumPlatform : public cudaq::quantum_platform {

public:
  ~MultiQPUQuantumPlatform() {
    // Make sure that we clean up the client QPUs first before cleaning up the
    // remote servers.
    platformQPUs.clear();
  }

  MultiQPUQuantumPlatform() { populateDefaultQPUs(); }

  bool supports_task_distribution() const override { return true; }

  void beginExecution() override {
    // Only set the CUDA device when GPU-backed QPUs are active.
    // Non-GPU platforms (e.g. ORCA) that replace the default QPUs
    // via setTargetBackend do not require a CUDA device assignment.
    auto qid = cudaq::getCurrentQpuId();
    int nDevices = cudaq::getCudaDeviceCount();
    if (nDevices > 0)
      cudaq::setCudaDevice(qid);
    // Base implementation of beginExecution will be called after this.
    cudaq::quantum_platform::beginExecution();
  }

private:
  void populateDefaultQPUs();

  static std::string getTargetName(const std::string &description) {
    // Target name is the first one in the target config string
    // or the whole string if this is the only config.
    return description.find(";") != std::string::npos
               ? cudaq::split(description, ';').front()
               : description;
  }

  static std::string getQpuType(const std::string &description) {
    // Target name is the first one in the target config string
    // or the whole string if this is the only config.
    const auto targetName = getTargetName(description);
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string targetConfigFileName = targetName + std::string(".yml");
    const auto explicitConfigPath =
        cudaq::detail::getBackendConfigOption(description, "__yml_path");
    auto configFilePath = explicitConfigPath
                              ? std::filesystem::path(*explicitConfigPath)
                              : platformPath / targetConfigFileName;
    CUDAQ_INFO("Config file path for target {} = {}", targetName,
               configFilePath.string());
    // Don't try to load something that doesn't exist.
    if (!explicitConfigPath && !std::filesystem::exists(configFilePath))
      return "";
    auto config = cudaq::config::loadTargetConfig(configFilePath);
    cudaq::detail::loadTargetPluginLibraries(targetName, configFilePath,
                                             config);

    if (config.BackendConfig.has_value() &&
        !config.BackendConfig->PlatformQpu.empty()) {
      return config.BackendConfig->PlatformQpu;
    }

    return "";
  }
  static std::string getOption(const std::string &str,
                               const std::string &prefix) {
    // Return the first key-value configuration option found in the format:
    // "<prefix>;<option>".
    // Note: This expects an exact match of the prefix and the option value is
    // the next one.
    return cudaq::detail::getBackendConfigOption(str, prefix).value_or("");
  }

  static std::string formatUrl(const std::string &url) {
    auto formatted = url;
    // Default to http:// if none provided.
    if (!formatted.starts_with("http"))
      formatted = std::string("http://") + formatted;
    if (!formatted.empty() && formatted.back() != '/')
      formatted += '/';
    return formatted;
  }

  void setTargetBackend(const std::string &description) override {
    const auto qpuSubType = getQpuType(description);
    if (!qpuSubType.empty()) {
      if (!cudaq::registry::isRegistered<cudaq::QPU>(qpuSubType))
        throw std::runtime_error(
            fmt::format("Unable to retrieve {} QPU implementation. Please "
                        "check your installation.",
                        qpuSubType));
      if (qpuSubType == "orca") {
        auto urls = cudaq::split(getOption(description, "url"), ',');
        platformQPUs.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          // Populate the information and add the QPUs
          platformQPUs.emplace_back(cudaq::registry::get<cudaq::QPU>("orca"));
          platformQPUs.back()->setId(qId);
          const std::string configStr =
              fmt::format("orca;url;{}", formatUrl(urls[qId]));
          platformQPUs.back()->setTargetBackend(configStr);
        }
        return;
      } else {
        throw std::runtime_error(
            fmt::format("Unsupported platform QPU sub-type '{}' specified in "
                        "target config. Currently only 'orca' is supported.",
                        qpuSubType));
      }
    } else {
      populateDefaultQPUs();

      if (platformQPUs.empty()) {
        // No QPU (GPU simulator nor specified platform QPU) was able to be
        // initialized, so we can't run.
        throw std::runtime_error(
            "No platform QPU implementations available. Please check your "
            "installation and target configuration.");
      }
    }
  }
};

void MultiQPUQuantumPlatform::populateDefaultQPUs() {
  platformQPUs.clear();
  int nDevices = cudaq::getCudaDeviceCount();
  // Skipped if CUDA-Q was built with CUDA but no devices present at
  // runtime.
  if (nDevices > 0) {
    const char *envVal = std::getenv("CUDAQ_MQPU_NGPUS");
    if (envVal != nullptr) {
      int specifiedNDevices = 0;
      try {
        specifiedNDevices = std::stoi(envVal);
      } catch (...) {
        throw std::runtime_error("Invalid CUDAQ_MQPU_NGPUS environment "
                                 "variable, must be integer.");
      }

      if (specifiedNDevices < nDevices)
        nDevices = specifiedNDevices;
    }

    if (nDevices == 0)
      throw std::runtime_error("No GPUs available to instantiate platform.");

    // Add a QPU for each GPU.
    for (int i = 0; i < nDevices; i++) {
      platformQPUs.emplace_back(std::make_unique<cudaq::DefaultQPU>());
      platformQPUs.back()->setId(i);
    }
  }
}
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
