/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRemoteSimulatorQPU.h"

#include <fstream>

using namespace mlir;

namespace {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {

public:
  RemoteSimulatorQPU() : BaseRemoteSimulatorQPU() {
    m_mlirContext = cudaq::initializeMLIR();
  }

  RemoteSimulatorQPU(RemoteSimulatorQPU &&) = delete;
  virtual ~RemoteSimulatorQPU() = default;
};

/// Implementation of QPU subtype that submits simulation request to NVCF.
class NvcfSimulatorQPU : public RemoteSimulatorQPU {
public:
  NvcfSimulatorQPU() : RemoteSimulatorQPU() {
    m_client = cudaq::registry::get<cudaq::RemoteRuntimeClient>("NVCF");
  }

  // Encapsulates Nvcf configurations that we need.
  // Empty strings mean no config available.
  struct NvcfConfig {
    std::string apiKey;
    std::string functionId;
    std::string versionId;
  };

  virtual void setTargetBackend(const std::string &backend) override {
    auto parts = cudaq::split(backend, ';');
    if (parts.size() % 2 != 0)
      throw std::invalid_argument("Unexpected backend configuration string. "
                                  "Expecting a ';'-separated key-value pairs.");
    std::string apiKey, functionId, versionId;

    for (std::size_t i = 0; i < parts.size(); i += 2) {
      if (parts[i] == "simulator")
        m_simName = parts[i + 1];
      // First, check if api key or function Id is provided as target options.
      if (parts[i] == "function_id")
        functionId = parts[i + 1];
      if (parts[i] == "api_key")
        apiKey = parts[i + 1];
      if (parts[i] == "version_id")
        versionId = parts[i + 1];
    }
    // If none provided, look for them in environment variables or the config
    // file.
    const auto config = searchNvcfConfig();
    if (apiKey.empty())
      apiKey = config.apiKey;
    if (functionId.empty())
      functionId = config.functionId;
    if (versionId.empty())
      versionId = config.versionId;

    // API key and function Id are required.
    if (apiKey.empty())
      throw std::runtime_error(
          "Cannot find NVCF API key. Please provide a valid API key.");

    if (!apiKey.starts_with("nvapi-"))
      std::runtime_error(
          "An invalid NVCF API key is provided. Please check your settings.");
    if (functionId.empty())
      throw std::runtime_error(
          "Cannot find NVCF Function ID. Please provide a valid Function ID.");

    std::unordered_map<std::string, std::string> clientConfigs{
        {"api-key", apiKey}, {"function-id", functionId}};
    if (!versionId.empty())
      clientConfigs.emplace("version-id", versionId);
    m_client->setConfig(clientConfigs);
  }

private:
  // Helper to search NVCF config from environment variable or config file.
  NvcfConfig searchNvcfConfig() {
    NvcfConfig config;
    // Search from environment variable
    if (auto apiKey = std::getenv("NVCF_API_KEY")) {
      const auto key = std::string(apiKey);
      config.apiKey = key;
    }

    if (auto funcIdEnv = std::getenv("NVCF_FUNCTION_ID"))
      config.functionId = std::string(funcIdEnv);

    if (auto versionIdEnv = std::getenv("NVCF_FUNCTION_VERSION_ID"))
      config.versionId = std::string(versionIdEnv);

    std::string nvcfConfig;
    // Allow someone to tweak this with an environment variable
    if (auto creds = std::getenv("CUDAQ_NVCF_CREDENTIALS"))
      nvcfConfig = std::string(creds);
    else
      nvcfConfig = std::string(getenv("HOME")) + std::string("/.nvcf_config");
    if (cudaq::fileExists(nvcfConfig)) {
      std::ifstream stream(nvcfConfig);
      std::string contents((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
      std::vector<std::string> lines;
      lines = cudaq::split(contents, '\n');
      for (const std::string &l : lines) {
        std::vector<std::string> keyAndValue = cudaq::split(l, ':');
        if (keyAndValue.size() != 2)
          throw std::runtime_error("Ill-formed configuration file (" +
                                   nvcfConfig +
                                   "). Key-value pairs must be in `<key> : "
                                   "<value>` format. (One per line)");
        cudaq::trim(keyAndValue[0]);
        cudaq::trim(keyAndValue[1]);
        if (config.apiKey.empty() &&
            (keyAndValue[0] == "key" || keyAndValue[0] == "apikey"))
          config.apiKey = keyAndValue[1];
        if (config.functionId.empty() && (keyAndValue[0] == "function-id" ||
                                          keyAndValue[0] == "Function ID"))
          config.functionId = keyAndValue[1];
        if (config.versionId.empty() &&
            (keyAndValue[0] == "version-id" || keyAndValue[0] == "Version ID"))
          config.versionId = keyAndValue[1];
      }
    }
    return config;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteSimulatorQPU, RemoteSimulatorQPU)
CUDAQ_REGISTER_TYPE(cudaq::QPU, NvcfSimulatorQPU, NvcfSimulatorQPU)
