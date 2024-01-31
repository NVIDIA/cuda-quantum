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
#include "helpers/MQPUUtils.h"
#include "llvm/Support/Base64.h"
#include <fstream>

LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

namespace {
class MultiQPUQuantumPlatform : public cudaq::quantum_platform {
  std::vector<std::unique_ptr<cudaq::AutoLaunchRestServerProcess>>
      m_remoteServers;

public:
  ~MultiQPUQuantumPlatform() = default;
  MultiQPUQuantumPlatform() {
    if (cudaq::registry::isRegistered<cudaq::QPU>("GPUEmulatedQPU")) {
      int nDevices = cudaq::getCudaGetDeviceCount();
      // Skipped if CUDAQ was built with CUDA but no devices present at runtime.
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

        platformNumQPUs = platformQPUs.size();
        platformCurrentQPU = 0;
      }
    }
  }

  bool supports_task_distribution() const override { return true; }

  void setTargetBackend(const std::string &description) override {
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
          cudaq::debug(
              "Retrieved option '{}' for the key '{}' from input string '{}'",
              splitParts[i + 1], prefix, str);
          if (splitParts[i + 1].starts_with("base64_")) {
            splitParts[i + 1].erase(0, 7); // erase "base64_"
            std::vector<char> decoded_vec;
            if (auto err = llvm::decodeBase64(splitParts[i + 1], decoded_vec))
              throw std::runtime_error("DecodeBase64 error");
            std::string decodedStr(decoded_vec.data(), decoded_vec.size());
            cudaq::info("Decoded {} parameter from '{}' to '{}'", splitParts[i],
                        splitParts[i + 1], decodedStr);
            return decodedStr;
          }
          return splitParts[i + 1];
        }
      }
      return "";
    };

    if (description.find("remote_execution") != std::string::npos) {
      if (!cudaq::registry::isRegistered<cudaq::QPU>("RemoteSimulatorQPU"))
        throw std::runtime_error(
            "Unable to retrieve RemoteSimulatorQPU implementation.");
      if (description.find("nvcf") != std::string::npos) {
        platformQPUs.clear();
        // Populate the information and add the QPUs
        auto qpu = cudaq::registry::get<cudaq::QPU>("RemoteSimulatorQPU");
        qpu->setId(0);
        auto simName = getOpt(description, "backend");
        if (simName.empty())
          simName = "custatevec-fp32";
        const std::string configStr =
            fmt::format("target;nvcf;simulator;{}", simName);
        qpu->setTargetBackend(configStr);
        threadToQpuId[std::hash<std::thread::id>{}(
            qpu->getExecutionThreadId())] = 0;
        platformQPUs.emplace_back(std::move(qpu));

        platformNumQPUs = platformQPUs.size();
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

        const auto formatUrl = [](const std::string &url) -> std::string {
          auto formatted = url;
          // Default to http:// if none provided.
          if (!formatted.starts_with("http"))
            formatted = std::string("http://") + formatted;
          if (!formatted.empty() && formatted.back() != '/')
            formatted += '/';
          return formatted;
        };
        if (autoLaunch) {
          urls.clear();
          const auto numInstanceStr = getOpt(description, "auto_launch");
          // Default to launching one instance if no other setting is available.
          const int numInstances =
              numInstanceStr.empty() ? 1 : std::stoi(numInstanceStr);
          cudaq::info("Auto launch {} REST servers", numInstances);
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
          threadToQpuId[std::hash<std::thread::id>{}(
              qpu->getExecutionThreadId())] = qId;
          platformQPUs.emplace_back(std::move(qpu));
        }
        platformNumQPUs = platformQPUs.size();
      }
    }
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
