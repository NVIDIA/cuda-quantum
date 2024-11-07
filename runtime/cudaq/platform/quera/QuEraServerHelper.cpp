/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraServerHelper.h"

namespace cudaq {

void QuEraServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing QuEra via Amazon Braket.");

  // Hard-coded for now
  auto machine = AQUILA;
  auto deviceArn = AQUILA_ARN;

  cudaq::info("Running on device {}", deviceArn);

  config["deviceArn"] = deviceArn;
  config["qubits"] = deviceQubitCounts.contains(deviceArn)
                         ? deviceQubitCounts.at(deviceArn)
                         : DEFAULT_QUBIT_COUNT;
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  parseConfigForCommonParams(config);

  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Create a job for the QuEra quantum computer
ServerJobPayload
QuEraServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  ServerJobPayload ret;
  std::vector<ServerMessage> &tasks = std::get<2>(ret);
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage taskRequest;
    taskRequest["name"] = circuitCode.name;
    taskRequest["deviceArn"] = backendConfig.at("deviceArn");

    /// FIXME: When plugged in with user-level API, may have to construct JSON
    /// (add headers etc.) here
    auto action = nlohmann::json::parse(circuitCode.code);
    taskRequest["action"] = action.dump();
    taskRequest["shots"] = shots;

    tasks.push_back(taskRequest);
  }

  cudaq::info("Created job payload for QuEra, targeting device {}",
              backendConfig.at("deviceArn"));

  return ret;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuEraServerHelper, quera)
