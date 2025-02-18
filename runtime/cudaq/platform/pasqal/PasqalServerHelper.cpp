/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/AnalogHamiltonian.h"
#include "common/Logger.h"
#include "PasqalServerHelper.h"

#include <unordered_map>
#include <unordered_set>

namespace cudaq {

void PasqalServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initialize Pasqal Cloud.");

  // Hard-coded for now.
  const std::string MACHINE = "Fresnel";
  const int MAX_QUBITS = 100;

  cudaq::info("Running on device {}", MACHINE);

  if (!config.contains("machine"))
    config["machine"] = MACHINE;
    
  config["qubits"] = MAX_QUBITS;

  if(!config["shots"].empty())
    setShots(std::stoul(config["shots"]));

  if (auto project_id = std::getenv("PASQAL_PROJECT_ID"))
    config["project_id"] = project_id;
    else
    config["project_id"] = "";

  parseConfigForCommonParams(config);

  backendConfig = std::move(config);
}

RestHeaders PasqalServerHelper::getHeaders() {
  std::string token;

  if (auto auth_token = std::getenv("PASQAL_AUTH_TOKEN"))
    token = "Bearer " + std::string(auth_token);
  else
    token = "Bearer ";

  std::map<std::string, std::string> headers{
    {"Authorization", token},
    {"Content-Type", "application/json"},
    {"User-Agent", "Cudaq/Pasqal"},
    {"Connection", "keep-alive"},
    {"Accept", "*/*"}};

  return headers;
}

ServerJobPayload
PasqalServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> tasks;

  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["name"] = circuitCode.name;
    message["machine"] = backendConfig.at("machine");
    message["shots"] = shots;
    message["project_id"] = backendConfig.at("project_id");

    auto sequence = nlohmann::json::parse(circuitCode.code);
    message["sequence"] = sequence.dump();

    tasks.push_back(message);
  }

  cudaq::info("Created job payload for Pasqal, targeting device {}",
              backendConfig.at("machine"));
  
  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(baseUrl + apiPath + "/v1/cudaq/job", getHeaders(), tasks);
}

std::string PasqalServerHelper::extractJobId(ServerMessage &postResponse) {
    return postResponse["data"]["id"].get<std::string>();
}

std::string PasqalServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + apiPath + "/v1/batches/" + jobId + "/results";
}

std::string
PasqalServerHelper::constructGetJobPath(ServerMessage &postResponse) {
    return baseUrl + apiPath + "/v1/batches/" +
      postResponse["data"]["id"].get<std::string>() + "/results";
}

bool PasqalServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string>
  terminals = {"PENDING", "RUNNING", "DONE", "ERROR", "CANCEL"};
  
  auto jobStatus = getJobResponse["data"]["status"].get<std::string>();
  return terminals.find(jobStatus) != terminals.end();
}

sample_result
PasqalServerHelper::processResults(ServerMessage &postJobResponse,
                                   std::string &jobId) {

  auto jobStatus = postJobResponse["data"]["status"].get<std::string>();
  if (jobStatus != "DONE") 
    throw std::runtime_error("Job status: " + jobStatus);

  std::vector<ExecutionResult> results;
  auto jobs = postJobResponse["data"]["jobs"];
  for (auto &job : jobs) {
    auto result = job["full_result"]["counter"].get<std::unordered_map<std::string, std::size_t>>();
    results.push_back(ExecutionResult(result));
  }

  // TODO: Check the index order.
  return sample_result(results);
}

} // namespace cudaq

// Register the Pasqal server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::PasqalServerHelper, pasqal)