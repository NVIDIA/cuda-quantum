/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <thread>

namespace cudaq {

class IonQServerHelper : public ServerHelper {
private:
  RestClient client;
  std::string getEnvVar(const std::string &key) const;
  bool keyExists(const std::string &key) const;

public:
  const std::string name() const override { return "ionq"; }
  RestHeaders getHeaders() override;
  void initialize(BackendConfig config) override;
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;
  std::string extractJobId(ServerMessage &postResponse) override;
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;
  std::string constructGetResultsPath(ServerMessage &postResponse);
  std::string constructGetResultsPath(std::string &jobId);
  ServerMessage getResults(std::string &resultsGetPath);
  bool jobIsDone(ServerMessage &getJobResponse) override;
  cudaq::sample_result processResults(ServerMessage &postJobResponse) override;
};

void IonQServerHelper::initialize(BackendConfig config) {
  backendConfig = std::move(config);
  backendConfig["url"] = "https://api.ionq.co";
  backendConfig["version"] = "v0.3";
  backendConfig["user_agent"] = "cudaq/0.3.0";
  backendConfig["target"] = "simulator";
  backendConfig["qubits"] = 29;
  backendConfig["token"] = getEnvVar("IONQ_API_KEY");
  backendConfig["job_path"] =
      backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";
}

std::string IonQServerHelper::getEnvVar(const std::string &key) const {
  const char *env_var = std::getenv(key.c_str());
  if (env_var == nullptr) {
    // Handle the case where the environment variable is not set
    throw std::runtime_error(key + " environment variable is not set.");
  }
  return std::string(env_var);
}

bool IonQServerHelper::keyExists(const std::string &key) const {
  // check if a key exists in the map
  return backendConfig.find(key) != backendConfig.end();
}

ServerJobPayload
IonQServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Check keys existence before accessing them
  if (!keyExists("target") || !keyExists("qubits") || !keyExists("job_path"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  ServerMessage job;
  job["target"] = backendConfig.at("target");
  job["qubits"] = backendConfig.at("qubits");
  job["shots"] = static_cast<int>(shots);
  job["input"] = {{"format", "qir"}, {"data", circuitCodes.front().code}};

  return std::make_tuple(backendConfig.at("job_path"), getHeaders(),
                         std::vector<ServerMessage>{job});
}

std::string IonQServerHelper::extractJobId(ServerMessage &postResponse) {
  if (!postResponse.contains("id"))
    throw std::runtime_error("ServerMessage doesn't contain 'id' key.");

  return postResponse.at("id");
}

std::string IonQServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  if (!postResponse.contains("results_url"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'results_url' key.");

  if (!keyExists("url"))
    throw std::runtime_error("Key 'url' doesn't exist in backendConfig.");

  return backendConfig.at("url") +
         postResponse.at("results_url").get<std::string>();
}

std::string IonQServerHelper::constructGetJobPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  return backendConfig.at("job_path") + "?id=" + jobId;
}

std::string
IonQServerHelper::constructGetResultsPath(ServerMessage &resultsGetPath) {
  if (!resultsGetPath.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = resultsGetPath.at("jobs");

  if (jobs.empty() || !jobs[0].contains("results_url"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'results_url' key in the first job.");

  if (!keyExists("url"))
    throw std::runtime_error("Key 'url' doesn't exist in backendConfig.");

  return backendConfig.at("url") + jobs[0].at("results_url").get<std::string>();
}

std::string IonQServerHelper::constructGetResultsPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  return backendConfig.at("job_path") + jobId + "/results";
}

ServerMessage IonQServerHelper::getResults(std::string &resultsGetPath) {
  auto headers = getHeaders();
  return client.get(resultsGetPath, "", headers);
}

bool IonQServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  if (!getJobResponse.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = getJobResponse.at("jobs");

  if (jobs.empty() || !jobs[0].contains("status"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'status' key in the first job.");

  return jobs[0].at("status").get<std::string>() == "completed";
}

cudaq::sample_result
IonQServerHelper::processResults(ServerMessage &postJobResponse) {
  auto resultsGetPath = constructGetResultsPath(postJobResponse);
  auto results = getResults(resultsGetPath);
  cudaq::CountsDictionary counts;

  for (const auto &element : results.items()) {
    std::string key = element.key();
    double value = element.value().get<double>();
    std::size_t count = static_cast<std::size_t>(value * shots);
    counts[key] = count;
  }

  // construct an ExecutionResult with the counts
  cudaq::ExecutionResult executionResult(counts);
  return cudaq::sample_result(executionResult);
}

RestHeaders IonQServerHelper::getHeaders() {
  if (!keyExists("token") || !keyExists("user_agent"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  RestHeaders headers;
  headers["Authorization"] = "apiKey " + backendConfig.at("token");
  headers["Content-Type"] = "application/json";
  headers["User-Agent"] = backendConfig.at("user_agent");
  return headers;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
