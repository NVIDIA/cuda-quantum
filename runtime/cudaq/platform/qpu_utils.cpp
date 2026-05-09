/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qpu_utils.h"
#include "common/Executor.h"
#include "common/RuntimeTarget.h"
#include "common/ServerHelper.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/Support/Base64.h"

using namespace cudaq;

void detail::parseTargetConfigYml(const std::string &yamlContent,
                                  config::TargetConfig &targetConfig) {
  llvm::yaml::Input Input(yamlContent.c_str());
  Input >> targetConfig;
}

std::string detail::decodeBase64(const std::string &encoded) {
  std::vector<char> decoded_vec;
  if (auto err = llvm::decodeBase64(encoded, decoded_vec))
    throw std::runtime_error("DecodeBase64 error");
  return std::string(decoded_vec.data(), decoded_vec.size());
}

bool detail::isAnalogHamiltonianKernel(const std::string &kernelName) {
  return kernelName.find(cudaq::runtime::cudaqAHKPrefixName) == 0;
}

void detail::initServerHelperAndExecutor(
    const std::string &qpuName,
    const std::map<std::string, std::string> &backendConfig,
    const config::TargetConfig &targetConfig,
    owning_ptr<ServerHelper> &serverHelper,
    std::unique_ptr<Executor> &executor) {
  // Create the ServerHelper for this QPU and give it the backend config.
  // The registry hands back a `unique_ptr<ServerHelper>` with the default
  // deleter; rebind it into an owning_ptr<ServerHelper> so that destruction
  // goes through the out-of-line `opaque_deleter<ServerHelper>` defined in
  // ServerHelper.cpp.
  auto raw = cudaq::registry::get<cudaq::ServerHelper>(qpuName);
  if (!raw) {
    throw std::runtime_error("ServerHelper not found for target: " + qpuName);
  }
  serverHelper = owning_ptr<ServerHelper>(raw.release());

  serverHelper->initialize(backendConfig);
  CUDAQ_INFO("Retrieving executor with name {}", qpuName);
  CUDAQ_INFO("Is this executor registered? {}",
             cudaq::registry::isRegistered<cudaq::Executor>(qpuName));
  executor = cudaq::registry::isRegistered<cudaq::Executor>(qpuName)
                 ? cudaq::registry::get<cudaq::Executor>(qpuName)
                 : std::make_unique<cudaq::Executor>();

  // Give the server helper to the executor
  executor->setServerHelper(serverHelper.get());

  // Construct the runtime target
  RuntimeTarget runtimeTarget;
  runtimeTarget.config = targetConfig;
  runtimeTarget.name = qpuName;
  runtimeTarget.description = targetConfig.Description;
  runtimeTarget.runtimeConfig = backendConfig;
  serverHelper->setRuntimeTarget(runtimeTarget);
}
