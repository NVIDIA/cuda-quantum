/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QDMIDevice.h"
#include "QDMIQPU.h"

#include "common/RuntimeTarget.h"
#include "qdmi/driver/Driver.hpp"
#include "cudaq/Target/TargetConfigYaml.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using BackendConfig = std::map<std::string, std::string>;
using QDMISite = typename decltype(std::declval<cudaq::FoMaCDevice &>()
                                       .getRegularSites())::value_type;

std::string decodeBackendValue(std::string value) {
  if (!value.starts_with("base64_"))
    return value;

  value.erase(0, 7);
  std::vector<char> decoded;
  if (auto err = llvm::decodeBase64(value, decoded))
    throw std::runtime_error("DecodeBase64 error");
  return std::string(decoded.data(), decoded.size());
}

BackendConfig parseBackendConfig(const std::string &description,
                                 std::string &targetName) {
  BackendConfig config;
  targetName = description;

  if (targetName.find(";") == std::string::npos)
    return config;

  auto split = cudaq::split(targetName, ';');
  targetName = split[0];
  if ((split.size() - 1) % 2 != 0)
    throw std::runtime_error(
        "Backend config must be provided as key-value pairs.");

  for (std::size_t i = 1; i < split.size(); i += 2)
    config.insert({split[i], decodeBackendValue(split[i + 1])});
  return config;
}

std::optional<std::string> getValue(const BackendConfig &config,
                                    const std::string &key,
                                    const char *envName) {
  if (auto iter = config.find(key);
      iter != config.end() && !iter->second.empty())
    return iter->second;
  if (const char *value = std::getenv(envName); value && value[0] != '\0')
    return std::string(value);
  return std::nullopt;
}

cudaq::config::TargetConfig loadTargetConfig(const std::string &targetName) {
  std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  const auto platformPath =
      cudaqLibPath.parent_path().parent_path() / "targets";
  const auto configFilePath = platformPath / (targetName + ".yml");
  CUDAQ_INFO("QDMI config file path = {}", configFilePath.string());

  std::ifstream configFile(configFilePath.string());
  if (!configFile)
    throw std::runtime_error("Could not open QDMI target configuration.");

  const std::string configContents((std::istreambuf_iterator<char>(configFile)),
                                   std::istreambuf_iterator<char>());
  cudaq::config::TargetConfig config;
  llvm::yaml::Input input(configContents.c_str());
  input >> config;
  return config;
}

qdmi::DeviceSessionConfig makeDeviceSessionConfig(const BackendConfig &config) {
  qdmi::DeviceSessionConfig sessionConfig;
  sessionConfig.baseUrl =
      getValue(config, "qdmi_base_url", "CUDAQ_QDMI_BASE_URL");
  sessionConfig.token = getValue(config, "qdmi_token", "CUDAQ_QDMI_TOKEN");
  sessionConfig.authFile =
      getValue(config, "qdmi_auth_file", "CUDAQ_QDMI_AUTH_FILE");
  sessionConfig.authUrl =
      getValue(config, "qdmi_auth_url", "CUDAQ_QDMI_AUTH_URL");
  sessionConfig.username =
      getValue(config, "qdmi_username", "CUDAQ_QDMI_USERNAME");
  sessionConfig.password =
      getValue(config, "qdmi_password", "CUDAQ_QDMI_PASSWORD");
  return sessionConfig;
}

fomac::SessionConfig makeFoMaCSessionConfig(const BackendConfig &config) {
  fomac::SessionConfig sessionConfig;
  sessionConfig.token = getValue(config, "qdmi_token", "CUDAQ_QDMI_TOKEN");
  sessionConfig.authFile =
      getValue(config, "qdmi_auth_file", "CUDAQ_QDMI_AUTH_FILE");
  sessionConfig.authUrl =
      getValue(config, "qdmi_auth_url", "CUDAQ_QDMI_AUTH_URL");
  sessionConfig.username =
      getValue(config, "qdmi_username", "CUDAQ_QDMI_USERNAME");
  sessionConfig.password =
      getValue(config, "qdmi_password", "CUDAQ_QDMI_PASSWORD");
  return sessionConfig;
}

std::set<QDMI_Device> getDeviceHandles(fomac::Session &session) {
  std::set<QDMI_Device> handles;
  for (const auto &device : session.getDevices())
    handles.insert(static_cast<QDMI_Device>(device));
  return handles;
}

std::vector<cudaq::FoMaCDevice>
loadDevices(const BackendConfig &config,
            std::unique_ptr<fomac::Session> &session) {
  const auto library = getValue(config, "qdmi_library", "CUDAQ_QDMI_LIBRARY");
  const auto prefix = getValue(config, "qdmi_prefix", "CUDAQ_QDMI_PREFIX");
  if (!library)
    throw std::runtime_error("QDMI device library is required.");
  if (!prefix)
    throw std::runtime_error("QDMI function prefix is required.");

  auto sessionConfig = makeFoMaCSessionConfig(config);
  std::set<QDMI_Device> oldDevices;
  {
    fomac::Session previousSession(sessionConfig);
    oldDevices = getDeviceHandles(previousSession);
  }

  const auto rootDevice = qdmi::Driver::get().addDynamicDeviceLibrary(
      *library, *prefix, makeDeviceSessionConfig(config));

  session = std::make_unique<fomac::Session>(sessionConfig);
  auto devices = session->getDevices();
  std::vector<cudaq::FoMaCDevice> selectedDevices;
  std::set<QDMI_Device> selectedHandles;

  for (auto &device : devices) {
    const auto handle = static_cast<QDMI_Device>(device);
    if (oldDevices.contains(handle))
      continue;
    if (selectedHandles.insert(handle).second)
      selectedDevices.emplace_back(std::move(device));
  }

  if (selectedDevices.empty()) {
    for (auto &device : devices) {
      const auto handle = static_cast<QDMI_Device>(device);
      if (handle == rootDevice && selectedHandles.insert(handle).second)
        selectedDevices.emplace_back(std::move(device));
    }
  }

  if (selectedDevices.empty())
    throw std::runtime_error("No QDMI devices were discovered.");

  CUDAQ_INFO("Loaded QDMI device library '{}'.", *library);
  return selectedDevices;
}

QDMI_Program_Format
selectProgramFormat(const std::vector<QDMI_Program_Format> &formats) {
  auto supports = [&formats](QDMI_Program_Format format) {
    return std::find(formats.begin(), formats.end(), format) != formats.end();
  };

  if (supports(QDMI_PROGRAM_FORMAT_QASM2))
    return QDMI_PROGRAM_FORMAT_QASM2;

  throw std::runtime_error("CUDA-Q QDMI target requires QASM2 support.");
}

std::map<std::size_t, std::size_t>
getRegularSitePositions(const cudaq::FoMaCDevice &device) {
  std::map<std::size_t, std::size_t> positions;
  try {
    const auto sites = device.getRegularSites();
    for (std::size_t index = 0; const auto &site : sites)
      positions.emplace(site.getIndex(), index++);
  } catch (const std::exception &e) {
    CUDAQ_DBG("QDMI regular site metadata is unavailable: {}", e.what());
  }
  return positions;
}

std::optional<std::size_t>
getQubitIndex(const QDMISite &site,
              const std::map<std::size_t, std::size_t> &positions,
              std::size_t qubitCount) {
  const auto siteIndex = site.getIndex();
  if (auto iter = positions.find(siteIndex); iter != positions.end())
    return iter->second;
  if (siteIndex < qubitCount)
    return siteIndex;
  return std::nullopt;
}

std::optional<std::vector<std::pair<std::size_t, std::size_t>>>
queryConnectivity(const cudaq::FoMaCDevice &device, std::size_t qubitCount) {
  const auto positions = getRegularSitePositions(device);
  std::set<std::pair<std::size_t, std::size_t>> edges;

  auto addEdge = [&](const QDMISite &sourceSite, const QDMISite &targetSite) {
    auto source = getQubitIndex(sourceSite, positions, qubitCount);
    auto target = getQubitIndex(targetSite, positions, qubitCount);
    if (!source || !target || *source == *target)
      return;
    edges.emplace(std::min(*source, *target), std::max(*source, *target));
  };

  if (auto couplingMap = device.getCouplingMap()) {
    for (const auto &[source, target] : *couplingMap)
      addEdge(source, target);
    return std::vector<std::pair<std::size_t, std::size_t>>(edges.begin(),
                                                            edges.end());
  }

  try {
    for (const auto &operation : device.getOperations()) {
      if (auto sitePairs = operation.getSitePairs()) {
        for (const auto &[source, target] : *sitePairs)
          addEdge(source, target);
      }
    }
  } catch (const std::exception &e) {
    CUDAQ_DBG("QDMI operation connectivity metadata is unavailable: {}",
              e.what());
  }

  if (edges.empty())
    return std::nullopt;
  return std::vector<std::pair<std::size_t, std::size_t>>(edges.begin(),
                                                          edges.end());
}

std::shared_ptr<cudaq::QDMIDevice> makeQDMIDevice(cudaq::FoMaCDevice device) {
  auto qdmiDevice = std::make_shared<cudaq::QDMIDevice>(std::move(device));
  qdmiDevice->name = qdmiDevice->device.getName();
  qdmiDevice->programFormat =
      selectProgramFormat(qdmiDevice->device.getSupportedProgramFormats());
  qdmiDevice->qubitCount = qdmiDevice->device.getQubitsNum();
  qdmiDevice->connectivity =
      queryConnectivity(qdmiDevice->device, qdmiDevice->qubitCount);

  CUDAQ_INFO("Discovered QDMI device '{}' with {} qubits.", qdmiDevice->name,
             qdmiDevice->qubitCount);
  return qdmiDevice;
}

class QDMIQuantumPlatform : public cudaq::quantum_platform {
public:
  ~QDMIQuantumPlatform() override {
    platformQPUs.clear();
    session.reset();
  }

  bool supports_task_distribution() const override {
    return platformQPUs.size() > 1;
  }

private:
  void setTargetBackend(const std::string &description) override {
    CUDAQ_INFO("QDMI target backend string is {}.", description);

    std::string targetName;
    auto backendConfig = parseBackendConfig(description, targetName);
    if (auto iter = backendConfig.find("emulate");
        iter != backendConfig.end() && iter->second == "true")
      throw std::runtime_error(
          "QDMI backend does not support CUDA-Q emulation mode.");

    cudaq::config::TargetConfig targetConfig;
    if (runtimeTarget) {
      targetConfig = runtimeTarget->config;
      runtimeTarget->runtimeConfig = backendConfig;
    } else {
      targetConfig = loadTargetConfig(targetName);
      runtimeTarget = std::make_unique<cudaq::RuntimeTarget>();
      runtimeTarget->name = targetName;
      runtimeTarget->platformName = "qdmi";
      runtimeTarget->description = targetConfig.Description;
      runtimeTarget->config = targetConfig;
      runtimeTarget->runtimeConfig = backendConfig;
    }

    platformQPUs.clear();
    session.reset();

    auto devices = loadDevices(backendConfig, session);
    for (std::size_t qpuId = 0; auto &device : devices) {
      platformQPUs.emplace_back(std::make_unique<cudaq::QDMIQPU>(
          makeQDMIDevice(std::move(device)), targetConfig, backendConfig));
      platformQPUs.back()->setId(qpuId++);
    }
  }

  std::unique_ptr<fomac::Session> session;
};
} // namespace

CUDAQ_REGISTER_PLATFORM(QDMIQuantumPlatform, qdmi)
