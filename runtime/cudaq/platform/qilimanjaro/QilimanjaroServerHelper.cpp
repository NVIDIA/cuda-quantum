/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QilimanjaroServerHelper.h"
#include "common/AnalogHamiltonian.h"
#include "common/Logger.h"

#include <unordered_map>
#include <unordered_set>

namespace cudaq {

void QilimanjaroServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initialize Qilimanjaro's SpeQtrum.");

  // Hard-coded for now.
  const std::string MACHINE = "radagast";

  cudaq::info("Running on device {}", MACHINE);

  if (!config.contains("machine"))
    config["machine"] = MACHINE;

  if (!config["nshots"].empty())
    setShots(std::stoul(config["nshots"]));

  parseConfigForCommonParams(config);

  backendConfig = std::move(config);
}

RestHeaders QilimanjaroServerHelper::getHeaders() {
  std::string token;

  if (auto auth_token = std::getenv("QILIMANJARO_AUTH_TOKEN"))
    token = "Bearer " + std::string(auth_token);
  else
    token = "Bearer ";

  std::map<std::string, std::string> headers{
      {"Authorization", token},
      {"Content-Type", "application/json"},
      {"User-Agent", "cudaq/0.12.0"},  // TODO: How to get version dynamically?
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};

  return headers;
}

ServerJobPayload
QilimanjaroServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> tasks;
  // TODO: circuitCodes needs to change to the Time Evolution JSON
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["device_code"] = backendConfig.at("machine");
    message["shots"] = shots;
    message["job_type"] = "analog";
    message["payload"] = nlohmann::json::parse(circuitCode.code);
    tasks.push_back(message);
  }

  cudaq::info("Created job payload for Qilimanjaro, targeting device {}",
              backendConfig.at("machine"));

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(speqtrumApiUrl + "/execute", getHeaders(), tasks);
}

std::string QilimanjaroServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string QilimanjaroServerHelper::constructGetJobPath(std::string &jobId) {
  return speqtrumApiUrl + "/jobs/" + jobId + "?result=true";
}

std::string
QilimanjaroServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  auto jobId = extractJobId(postResponse);
  return constructGetJobPath(jobId);
}

bool QilimanjaroServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string> terminal_states = {"completed", "error", "canceled", "timeout"};
  auto jobStatus = getJobResponse["status"].get<std::string>();
  return terminal_states.find(jobStatus) != terminal_states.end();
}

sample_result QilimanjaroServerHelper::processResults(ServerMessage &postJobResponse,
                                                      std::string &jobId) {
  auto jobStatus = postJobResponse["status"].get<std::string>();
  if (jobStatus != "completed")
    throw std::runtime_error("Job status: " + jobStatus);
  
  auto job_result = postJobResponse["result"];
  
  // TODO: We need a new method signature with EvolveResults instead.
  std::vector<ExecutionResult> results;

  return sample_result(results);
}

} // namespace cudaq

// Register the Qilimanjaro server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QilimanjaroServerHelper, qilimanjaro)