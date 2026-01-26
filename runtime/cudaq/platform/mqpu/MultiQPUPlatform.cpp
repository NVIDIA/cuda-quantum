/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/NoiseModel.h"
#include "common/RuntimeTarget.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "helpers/MQPUUtils.h"
#include "utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <filesystem>
#include <fstream>

LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

namespace {
class MultiQPUQuantumPlatform : public cudaq::quantum_platform {
  std::vector<std::unique_ptr<cudaq::AutoLaunchRestServerProcess>>
      m_remoteServers;

public:
  ~MultiQPUQuantumPlatform() {
    // Make sure that we clean up the client QPUs first before cleaning up the
    // remote servers.
    platformQPUs.clear();
    m_remoteServers.clear();
  }

  MultiQPUQuantumPlatform() {
    if (cudaq::registry::isRegistered<cudaq::QPU>("GPUEmulatedQPU")) {
      int nDevices = cudaq::getCudaGetDeviceCount();
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
          throw std::runtime_error(
              "No GPUs available to instantiate platform.");

        // Add a QPU for each GPU.
        for (int i = 0; i < nDevices; i++) {
          platformQPUs.emplace_back(
              cudaq::registry::get<cudaq::QPU>("GPUEmulatedQPU"));
          platformQPUs.back()->setId(i);
        }
      }
    }
  }

  bool supports_task_distribution() const override { return true; }

  std::string getQpuType(const std::string &description) const {
    // Target name is the first one in the target config string
    // or the whole string if this is the only config.
    const auto targetName = description.find(";") != std::string::npos
                                ? cudaq::split(description, ';').front()
                                : description;
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string targetConfigFileName = targetName + std::string(".yml");
    auto configFilePath = platformPath / targetConfigFileName;
    CUDAQ_INFO("Config file path for target {} = {}", targetName,
               configFilePath.string());
    // Don't try to load something that doesn't exist.
    if (!std::filesystem::exists(configFilePath))
      return "";
    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;

    if (config.BackendConfig.has_value() &&
        !config.BackendConfig->PlatformQpu.empty()) {
      return config.BackendConfig->PlatformQpu;
    }

    return "";
  }

  void setTargetBackend(const std::string &description) override {
    executionContext.set(nullptr);

    const auto getOpt = [](const std::string &str,
                           const std::string &prefix) -> std::string {
      // Return the first key-value configuration option found in the format:
      // "<prefix>;<option>".
      // Note: This expects an exact match of the prefix and the option value is
      // the next one.
      auto splitParts = cudaq::split(str, ';');
      if (splitParts.empty())
        return "";
      for (std::size_t i = 0; i < splitParts.size() - 1; ++i) {
        if (splitParts[i] == prefix) {
          CUDAQ_DBG(
              "Retrieved option '{}' for the key '{}' from input string '{}'",
              splitParts[i + 1], prefix, str);
          if (splitParts[i + 1].starts_with("base64_")) {
            splitParts[i + 1].erase(0, 7); // erase "base64_"
            std::vector<char> decoded_vec;
            if (auto err = llvm::decodeBase64(splitParts[i + 1], decoded_vec))
              throw std::runtime_error("DecodeBase64 error");
            std::string decodedStr(decoded_vec.data(), decoded_vec.size());
            CUDAQ_INFO("Decoded {} parameter from '{}' to '{}'", splitParts[i],
                       splitParts[i + 1], decodedStr);
            return decodedStr;
          }
          return splitParts[i + 1];
        }
      }
      return "";
    };

    const auto qpuSubType = getQpuType(description);
    if (!qpuSubType.empty()) {
      const auto formatUrl = [](const std::string &url) -> std::string {
        auto formatted = url;
        // Default to http:// if none provided.
        if (!formatted.starts_with("http"))
          formatted = std::string("http://") + formatted;
        if (!formatted.empty() && formatted.back() != '/')
          formatted += '/';
        return formatted;
      };

      if (!cudaq::registry::isRegistered<cudaq::QPU>(qpuSubType))
        throw std::runtime_error(
            fmt::format("Unable to retrieve {} QPU implementation. Please "
                        "check your installation.",
                        qpuSubType));
      if (qpuSubType == "orca") {
        auto urls = cudaq::split(getOpt(description, "url"), ',');
        platformQPUs.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          // Populate the information and add the QPUs
          platformQPUs.emplace_back(cudaq::registry::get<cudaq::QPU>("orca"));
          platformQPUs.back()->setId(qId);
          const std::string configStr =
              fmt::format("orca;url;{}", formatUrl(urls[qId]));
          platformQPUs.back()->setTargetBackend(configStr);
        }
      } else {
        auto urls = cudaq::split(getOpt(description, "url"), ',');
        auto sims = cudaq::split(getOpt(description, "backend"), ',');
        // Default to qpp simulator if none provided.
        if (sims.empty())
          sims.emplace_back("qpp");
        // If no URL is provided, default to auto launching one server instance.
        const bool autoLaunch =
            description.find("auto_launch") != std::string::npos ||
            urls.empty();

        if (autoLaunch) {
          urls.clear();
          const auto numInstanceStr = getOpt(description, "auto_launch");
          // Default to launching one instance if no other setting is available.
          const int numInstances =
              numInstanceStr.empty() ? 1 : std::stoi(numInstanceStr);
          CUDAQ_INFO("Auto launch {} REST servers", numInstances);
          for (int i = 0; i < numInstances; ++i) {
            m_remoteServers.emplace_back(
                std::make_unique<cudaq::AutoLaunchRestServerProcess>(i));
            urls.emplace_back(m_remoteServers.back()->getUrl());
          }
        }

        // List of simulator names must either be one or the same length as the
        // URL list. If one simulator name is provided, assuming that all the
        // URL should be using the same simulator.
        if (sims.size() > 1 && sims.size() != urls.size())
          throw std::runtime_error(fmt::format(
              "Invalid number of remote backend simulators provided: "
              "receiving {}, expecting {}.",
              sims.size(), urls.size()));
        platformQPUs.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          const auto simName = sims.size() == 1 ? sims.front() : sims[qId];
          // Populate the information and add the QPUs
          auto qpu = cudaq::registry::get<cudaq::QPU>("RemoteSimulatorQPU");
          qpu->setId(qId);
          const std::string configStr =
              fmt::format("url;{};simulator;{}", formatUrl(urls[qId]), simName);
          qpu->setTargetBackend(configStr);
          platformQPUs.emplace_back(std::move(qpu));
        }
      }
    }
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
