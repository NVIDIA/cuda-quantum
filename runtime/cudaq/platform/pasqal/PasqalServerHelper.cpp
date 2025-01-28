/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalServerHelper.h"
#include "common/AnalogHamiltonian.h"

namespace cudaq {

void PasqalServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing pasqal cloud");

  // Hard-coded for now
  const std::string FRESNEL = "fresnel1";
  auto machine = FRESNEL;
  const int MAX_QUBITS = 100;

  cudaq::info("Running on device {}", machine);
  config["machine"] = machine;
  config["qubits"] = MAX_QUBITS;

  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  parseConfigForCommonParams(config);

  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Get the headers for the API requests
RestHeaders PasqalServerHelper::getHeaders() { 
  std::string token, refreshKey, timeStr;
  if (auto auth_token = std::getenv("PASQAL_AUTH_TOKEN"))
    token = "Bearer " + std::string(auth_token);
  else
    token = "Bearer ";

  std::map<std::string, std::string> headers{
      {"Authorization", token},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
 }

// Create a job for the pasqal quantum computer
ServerJobPayload
PasqalServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  ServerJobPayload ret;
  std::vector<ServerMessage> &tasks = std::get<2>(ret);
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage taskRequest;

    taskRequest["name"] = circuitCode.name;
    taskRequest["device"] = backendConfig.at("machine");
    auto action = nlohmann::json::parse(circuitCode.code);
    taskRequest["action"] = action.dump();
    taskRequest["shots"] = shots;

    tasks.push_back(taskRequest);
  }
  cudaq::info("Created job payload for Pasqal, targeting device {}", backendConfig.at("machine"));
  // Return a tuple containing the job path, headers, and the job message
  const std::string baseUrl = "https://pasqal.cloud";
  return std::make_tuple(baseUrl + "v1/batches", getHeaders(), tasks);
}

} // namespace cudaq

// Register the Pasqal server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::PasqalServerHelper, pasqal)
