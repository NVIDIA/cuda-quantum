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
  const std::string MACHINE = "CUDA_DYNAMICS";

  cudaq::info("Running on device {}", MACHINE);

  if (!config.contains("machine"))
    config["machine"] = MACHINE;

  if (!config["shots"].empty())
    setShots(std::stoul(config["shots"]));

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
      {"User-Agent", "cudaq/0.12.0"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};

  return headers;
}

ServerJobPayload
QilimanjaroServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> tasks;

  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["machine"] = backendConfig.at("machine");
    message["shots"] = shots;
    message["project_id"] = backendConfig.at("project_id");
    message["sequence"] = nlohmann::json::parse(circuitCode.code);
    tasks.push_back(message);
  }

  cudaq::info("Created job payload for Qilimanjaro, targeting device {}",
              backendConfig.at("machine"));

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(speqtrumApiUrl + "/execute", getHeaders(), tasks);
}

std::string QilimanjaroServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["data"]["id"].get<std::string>();
}

std::string QilimanjaroServerHelper::constructGetJobPath(std::string &jobId) {
  return speqtrumApiUrl + "/jobs/" + jobId;
}

std::string
QilimanjaroServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return speqtrumApiUrl + "/jobs/" + postResponse["data"]["id"].get<std::string>();
}

bool QilimanjaroServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string> terminal_states = {"COMPLETED", "ERROR", "CANCELED"};
  auto jobStatus = getJobResponse["data"]["status"].get<std::string>();
  return terminal_states.find(jobStatus) != terminal_states.end();
}

sample_result QilimanjaroServerHelper::processResults(ServerMessage &postJobResponse,
                                                 std::string &jobId) {
  auto status = postJobResponse["data"]["status"].get<std::string>();
  if (status != "COMPLETED")
    throw std::runtime_error("Job status: " + status);

  std::vector<ExecutionResult> results;
  auto jobs = postJobResponse["data"]["result"];
  for (auto &job : jobs) {
    std::unordered_map<std::string, std::size_t> result;
    for (auto &[bitstring, count] : job.items()) {
      auto r_bitstring = bitstring;
      result[r_bitstring] = count;
    }
    results.push_back(ExecutionResult(result));
  }

  return sample_result(results);
}

} // namespace cudaq

// Register the Qilimanjaro server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QilimanjaroServerHelper, qilimanjaro)