/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QDMIPlatformDevice.h"
#include "QDMIQPU.h"

#include "common/RuntimeTarget.h"
#include "qdmi/driver/Driver.hpp"
#include "cudaq/Target/TargetConfigYaml.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"

#include <algorithm>
#include <array>
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
    config.insert({split[i], split[i + 1]});
  return config;
}

std::optional<std::string> getValue(const BackendConfig &config,
                                    const std::string &key) {
  if (auto iter = config.find(key);
      iter != config.end() && !iter->second.empty())
    return iter->second;
  return std::nullopt;
}

std::optional<fomac::CustomJobParameter>
getJobParameter(const BackendConfig &config, const std::string &key) {
  if (auto value = getValue(config, key))
    return fomac::CustomJobParameter{std::move(*value)};
  return std::nullopt;
}

BackendConfig resolveDeviceConfig(const BackendConfig &config) {
  constexpr std::array deviceOptions{
      "qdmi_library",         "qdmi_prefix",          "qdmi_base_url",
      "qdmi_token",           "qdmi_auth_file",       "qdmi_auth_url",
      "qdmi_username",        "qdmi_password",        "qdmi_session_custom1",
      "qdmi_session_custom2", "qdmi_session_custom3", "qdmi_session_custom4",
      "qdmi_session_custom5",
  };

  BackendConfig resolved;
  for (const auto *key : deviceOptions) {
    if (const auto value = getValue(config, key))
      resolved.emplace(key, *value);
  }
  return resolved;
}

cudaq::config::TargetConfig loadTargetConfig(const std::string &targetName) {
  std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  const auto platformPath =
      cudaqLibPath.parent_path().parent_path() / "targets";
  const auto configFilePath = platformPath / (targetName + ".yml");
  CUDAQ_INFO("QDMI config file path = {}", configFilePath.string());

  std::ifstream configFile(configFilePath.string());
  if (!configFile)
    throw std::runtime_error("Could not open QDMI target configuration '" +
                             configFilePath.string() + "'.");

  const std::string configContents((std::istreambuf_iterator<char>(configFile)),
                                   std::istreambuf_iterator<char>());
  cudaq::config::TargetConfig config;
  cudaq::detail::parseTargetConfigYml(configContents, config);
  return config;
}

qdmi::DeviceSessionConfig makeDeviceSessionConfig(const BackendConfig &config) {
  qdmi::DeviceSessionConfig sessionConfig;
  sessionConfig.baseUrl = getValue(config, "qdmi_base_url");
  sessionConfig.token = getValue(config, "qdmi_token");
  sessionConfig.authFile = getValue(config, "qdmi_auth_file");
  sessionConfig.authUrl = getValue(config, "qdmi_auth_url");
  sessionConfig.username = getValue(config, "qdmi_username");
  sessionConfig.password = getValue(config, "qdmi_password");
  sessionConfig.custom1 = getValue(config, "qdmi_session_custom1");
  sessionConfig.custom2 = getValue(config, "qdmi_session_custom2");
  sessionConfig.custom3 = getValue(config, "qdmi_session_custom3");
  sessionConfig.custom4 = getValue(config, "qdmi_session_custom4");
  sessionConfig.custom5 = getValue(config, "qdmi_session_custom5");
  return sessionConfig;
}

cudaq::FoMaCDevice loadDevice(const BackendConfig &config) {
  const auto library = getValue(config, "qdmi_library");
  const auto prefix = getValue(config, "qdmi_prefix");
  if (!library)
    throw std::runtime_error("QDMI device library is required.");
  if (!prefix)
    throw std::runtime_error("QDMI function prefix is required.");

  const auto rootDevice = qdmi::Driver::get().addDynamicDeviceLibrary(
      *library, *prefix, makeDeviceSessionConfig(config));

  CUDAQ_INFO("Loaded QDMI device library '{}'.", *library);
  return fomac::Session::createSessionlessDevice(rootDevice);
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
  if (auto iter = positions.find(siteIndex); iter != positions.end()) {
    if (iter->second < qubitCount)
      return iter->second;
    return std::nullopt;
  }
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

  const auto couplingMap = device.getCouplingMap();
  if (!couplingMap)
    return std::nullopt;

  for (const auto &[source, target] : *couplingMap)
    addEdge(source, target);
  return std::vector<std::pair<std::size_t, std::size_t>>(edges.begin(),
                                                          edges.end());
}

std::shared_ptr<cudaq::QDMIPlatformDevice>
makePlatformDevice(cudaq::FoMaCDevice device) {
  auto platformDevice =
      std::make_shared<cudaq::QDMIPlatformDevice>(std::move(device));
  platformDevice->name = platformDevice->fomacDevice.getName();

  platformDevice->programFormat = selectProgramFormat(
      platformDevice->fomacDevice.getSupportedProgramFormats());
  platformDevice->qubitCount = platformDevice->fomacDevice.getQubitsNum();
  platformDevice->connectivity = queryConnectivity(platformDevice->fomacDevice,
                                                   platformDevice->qubitCount);

  CUDAQ_INFO("Discovered QDMI device '{}' with {} qubits.",
             platformDevice->name, platformDevice->qubitCount);
  return platformDevice;
}

std::shared_ptr<cudaq::QDMIPlatformDevice> makeJobConfiguredDevice(
    const std::shared_ptr<cudaq::QDMIPlatformDevice> &device,
    const BackendConfig &config) {
  auto configuredDevice = std::make_shared<cudaq::QDMIPlatformDevice>(*device);
  configuredDevice->jobCustom1 = getJobParameter(config, "qdmi_job_custom1");
  configuredDevice->jobCustom2 = getJobParameter(config, "qdmi_job_custom2");
  configuredDevice->jobCustom3 = getJobParameter(config, "qdmi_job_custom3");
  configuredDevice->jobCustom4 = getJobParameter(config, "qdmi_job_custom4");
  configuredDevice->jobCustom5 = getJobParameter(config, "qdmi_job_custom5");
  return configuredDevice;
}

class QDMIQuantumPlatform : public cudaq::quantum_platform {
private:
  void setTargetBackend(const std::string &description) override {
    std::string targetName;
    auto backendConfig = parseBackendConfig(description, targetName);
    CUDAQ_INFO("QDMI platform is targeting '{}'.", targetName);
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

    const auto deviceConfig = resolveDeviceConfig(backendConfig);

    auto [deviceIter, inserted] =
        loadedPlatformDevices.try_emplace(deviceConfig);
    if (inserted) {
      try {
        deviceIter->second = makePlatformDevice(loadDevice(deviceConfig));
      } catch (...) {
        loadedPlatformDevices.erase(deviceIter);
        throw;
      }
    }

    platformQPUs.emplace_back(std::make_unique<cudaq::QDMIQPU>(
        makeJobConfiguredDevice(deviceIter->second, backendConfig),
        targetConfig, backendConfig));
    platformQPUs.back()->setId(0);
  }

  // MQT Core's driver owns loaded device libraries for the process lifetime
  // and does not expose an unregister operation. Keep one wrapper per resolved
  // device-session configuration to match that ownership model.
  std::map<BackendConfig, std::shared_ptr<cudaq::QDMIPlatformDevice>>
      loadedPlatformDevices;
};
} // namespace

CUDAQ_REGISTER_PLATFORM(QDMIQuantumPlatform, qdmi)
