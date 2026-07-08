/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "QDMIServerHelper.h"

#include "fomac/FoMaC.hpp"
#include "qdmi/driver/Driver.hpp"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::optional<std::string> getValue(const cudaq::BackendConfig &config,
                                    const std::string &key,
                                    const char *envName) {
  if (auto iter = config.find(key);
      iter != config.end() && !iter->second.empty())
    return iter->second;
  if (const char *value = std::getenv(envName); value && value[0] != '\0')
    return std::string(value);
  return std::nullopt;
}

QDMI_Program_Format
selectProgramFormat(const std::vector<QDMI_Program_Format> &formats) {
  auto supports = [&formats](QDMI_Program_Format format) {
    return std::find(formats.begin(), formats.end(), format) != formats.end();
  };

  if (supports(QDMI_PROGRAM_FORMAT_QASM2))
    return QDMI_PROGRAM_FORMAT_QASM2;

  throw std::runtime_error("CUDA-Q QDMI target requires a device that "
                           "supports QDMI_PROGRAM_FORMAT_QASM2.");
}

using QDMISite = fomac::Session::Device::Site;

std::map<std::size_t, std::size_t>
getRegularSitePositions(const fomac::Session::Device &device) {
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
queryConnectivity(const fomac::Session::Device &device,
                  std::size_t qubitCount) {
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

[[noreturn]] void reportRestUse() {
  throw std::runtime_error("QDMI backend uses direct device execution.");
}
} // namespace

namespace cudaq {

QDMIServerHelper::~QDMIServerHelper() = default;

void QDMIServerHelper::initialize(BackendConfig config) {
  backendConfig = config;
  parseConfigForCommonParams(config);

  if (auto iter = config.find("shots"); iter != config.end())
    setShots(std::stoul(iter->second));

  const auto library = getValue(config, "qdmi_library", "CUDAQ_QDMI_LIBRARY");
  const auto prefix = getValue(config, "qdmi_prefix", "CUDAQ_QDMI_PREFIX");
  if (!library)
    throw std::runtime_error("QDMI device library is required.");
  if (!prefix)
    throw std::runtime_error("QDMI function prefix is required.");

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

  device = qdmi::Driver::get().addDynamicDeviceLibrary(*library, *prefix,
                                                       sessionConfig);

  const auto qdmiDevice = fomac::Session::Device::fromQDMIDevice(device);
  programFormat = selectProgramFormat(qdmiDevice.getSupportedProgramFormats());
  qubitCount = qdmiDevice.getQubitsNum();
  connectivity = queryConnectivity(qdmiDevice, qubitCount);

  CUDAQ_INFO("Loaded QDMI device library '{}' with {} qubits.", *library,
             qubitCount);
}

ServerJobPayload QDMIServerHelper::createJob(std::vector<KernelExecution> &) {
  reportRestUse();
}

std::string QDMIServerHelper::extractJobId(ServerMessage &) { reportRestUse(); }

std::string QDMIServerHelper::constructGetJobPath(ServerMessage &) {
  reportRestUse();
}

std::string QDMIServerHelper::constructGetJobPath(std::string &) {
  reportRestUse();
}

bool QDMIServerHelper::jobIsDone(ServerMessage &) { reportRestUse(); }

sample_result QDMIServerHelper::processResults(ServerMessage &, std::string &) {
  reportRestUse();
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QDMIServerHelper, qdmi)
