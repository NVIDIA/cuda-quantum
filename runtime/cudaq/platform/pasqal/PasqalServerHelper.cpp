/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalServerHelper.h"
#include "PasqalUtils.h"
#include "common/AnalogHamiltonian.h"
#include "cudaq/runtime/logger/logger.h"

#include <cstdlib>
#include <unordered_set>

namespace cudaq {

void PasqalServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initialize Pasqal Cloud.");

  // Defaults
  const std::string MACHINE = "EMU_MPS";
  const int MAX_QUBITS = 100;

  if (!config.contains("machine"))
    config["machine"] = MACHINE;

  CUDAQ_INFO("Running on Pasqal machine {}", config["machine"]);

  config["qubits"] = MAX_QUBITS;

  if (!config["shots"].empty())
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
    message["machine"] = backendConfig.at("machine");
    message["shots"] = shots;
    message["project_id"] = backendConfig.at("project_id");
    message["sequence"] = nlohmann::json::parse(circuitCode.code);
    tasks.push_back(message);
  }

  CUDAQ_INFO("Created Pasqal payload, targeting machine {}",
             backendConfig.at("machine"));

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(baseUrl + apiPath + "/v1/cudaq/job", getHeaders(),
                         tasks);
}

std::string PasqalServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["data"]["id"].get<std::string>();
}

std::string PasqalServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + apiPath + "/v1/cudaq/job/" + jobId;
}

std::string
PasqalServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + apiPath + "/v1/cudaq/job/" +
         postResponse["data"]["id"].get<std::string>();
}

bool PasqalServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::unordered_set<std::string> terminals = {"DONE", "ERROR", "CANCELED",
                                               "TIMED_OUT", "PAUSED"};
  auto jobStatus = getJobResponse["data"]["status"].get<std::string>();
  return terminals.find(jobStatus) != terminals.end();
}

sample_result PasqalServerHelper::processResults(ServerMessage &postJobResponse,
                                                 std::string &jobId) {
  auto status = postJobResponse["data"]["status"].get<std::string>();
  if (status != "DONE")
    throw std::runtime_error("Job status: " + status);

  std::vector<ExecutionResult> results;
  auto jobs = postJobResponse["data"]["result"];

  // loop over jobs in batch to get results
  // Current implementation only has 1 job
  for (auto &job : jobs) {
    results.push_back(pasqal::parseExecutionResult(job));
  }

  return sample_result(results);
}
} // namespace cudaq

// Avoid duplicate "pasqal" registration from the Python dialect extension
// build; the runtime plugin provides the canonical registration.
#ifndef CUDAQuantumPythonModules_extension__quakeDialects_dso_EXPORTS
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::PasqalServerHelper, pasqal)
#endif
