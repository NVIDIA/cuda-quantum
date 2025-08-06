/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nlohmann/json.hpp"
#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <thread>

namespace cudaq {

/// @brief The QcIServerHelper class extends the ServerHelper class to handle
/// interactions with the QCI server for submitting and retrieving quantum
/// computation jobs.
class QCIServerHelper : public ServerHelper {
private:
  /// @brief Default API token. This is not a secret nor a credential. QCI uses
  /// these to identify the library or app and the version that is originating
  /// each network request to both monitor usage and to disable defective or
  /// obsolete releases. Each new release of the SDK should incorporate a new
  /// API token.
  const std::string DEFAULT_API_TOKEN =
      "eyJhbGciOiJFZDI1NTE5IiwidHlwIjoiSldUIn0."
      "eyJhdWQiOiJjbGllbnQiLCJleHAiOjIzNjUwOTQwMDQsImlhdCI6MTczNDM3NDAwNCwiaXNz"
      "IjoiYXF1bWVuIiwianRpIjoiMDE5M2QwYmYtMTY0OS03NTE4LWI3MTktNGJlNzU4MDRiNWNh"
      "IiwibmJmIjoxNzM0Mzc0MDA0LCJzdWIiOiIwMTkzZDBiZi0xNjQwLTdjYzQtOTU2Ni00YjJk"
      "M2ZjOTk2ZWYifQ._"
      "fdj5Mcsv8BAEuaIMfnIAHqme883wWZTfWCbN3zOodxxHjIL84B0CT9ULFRN3I_"
      "qUw4P3vmLM99f-tBu8hOKDw";

  /// @brief Default base URL for QCI's service.
  const std::string DEFAULT_API_URL =
      "https://beta-service-aqumen.sensedata.dev/";

  /// @brief Default machine, the simulator.
  const std::string DEFAULT_MACHINE = "simulator";

  /// @brief Polling interval for job status via QCI's CUDA-Q endpoint in
  /// microseconds.
  const std::size_t QCI_CUDAQ_ENDPOINT_POLL_MICRO = 1000000;

protected:
  /// @brief RestClient used to POST HTTP requests.
  RestClient restClient;

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeaders() const;

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key, const std::string &defaultVal,
                        const bool isRequired) const;

  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const;

public:
  /// @brief Returns the name of this backend.
  const std::string name() const override { return "qci"; }

  /// @brief Returns the HTTP headers for use with QCI's HTTP backend.
  RestHeaders getHeaders() override;

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override {
    CUDAQ_INFO("Initializing Quantum Circuits backend.");

    auto apiUrl = getEnvVar("QCI_API_URL", DEFAULT_API_URL, false);
    config["apiUrl"] = apiUrl.ends_with("/") ? apiUrl : apiUrl + "/";
    CUDAQ_INFO("QCI backend API URL: {}", config["apiUrl"]);

    config["apiToken"] = getEnvVar("QCI_API_TOKEN", DEFAULT_API_TOKEN, false);

    config["machine"] = getValueOrDefault(config, "machine", DEFAULT_MACHINE);
    CUDAQ_INFO("QCI backend machine: {}", config["machine"]);

    config["authToken"] =
        getEnvVar("QCI_AUTH_TOKEN", "QCI_AUTH_TOKEN_NOT_SET", true);

    if (!config["shots"].empty()) {
      this->setShots(std::stoul(config["shots"]));
      CUDAQ_INFO("QCI backend configured shots: {}", config["shots"]);
    }

    // Parse parameters common to all jobs, place into member variables, then
    // move into member variable backendConfig.
    parseConfigForCommonParams(config);
    backendConfig = std::move(config);
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Extracts the job ID from the server's response to a job
  /// submission.
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override;

  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

  /// @brief Retrieves the results of a job using the provided path.
  ServerMessage getResults(std::string &resultsGetPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;
};

// Retrieve an environment variable
std::string QCIServerHelper::getEnvVar(const std::string &key,
                                       const std::string &defaultVal,
                                       const bool isRequired) const {
  // Get the environment variable
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, either return the default or throw an
  // exception
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error(key + " environment variable is not set.");
    else
      return defaultVal;
  }
  // Return the variable as a string
  return std::string(env_var);
}

std::string
QCIServerHelper::getValueOrDefault(const BackendConfig &config,
                                   const std::string &key,
                                   const std::string &defaultValue) const {
  return config.find(key) != config.end() ? config.at(key) : defaultValue;
}

/// @brief Create job specifications for QCI's CUDA-Q endpoint.
ServerJobPayload
QCIServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;

  for (auto &circuitCode : circuitCodes) {
    ServerMessage job;
    job["code"] = circuitCode.code;
    job["machine"] = backendConfig.at("machine");
    job["mappingReorderIdx"] = circuitCode.mapping_reorder_idx;
    job["name"] = circuitCode.name;
    job["outputNames"] = circuitCode.output_names;
    job["shots"] = shots;
    job["userData"] = circuitCode.user_data;

    messages.push_back(job);
  }

  RestHeaders headers = generateRequestHeaders();
  return std::make_tuple(backendConfig.at("apiUrl") + "cudaq/v1/jobs", headers,
                         messages);
}

/// @brief Extract the job ID from the server's response to a job submission.
std::string QCIServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string QCIServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return backendConfig.at("apiUrl") + "cudaq/v1/jobs/" +
         extractJobId(postResponse);
}

std::string QCIServerHelper::constructGetJobPath(std::string &jobId) {
  return backendConfig.at("apiUrl") + "cudaq/v1/jobs/" + jobId;
}

bool QCIServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  CUDAQ_DBG("getJobResponse: {}", getJobResponse.dump());

  if (!getJobResponse["error"].is_null()) {
    std::string msg = getJobResponse["error"]["message"].get<std::string>();
    throw std::runtime_error("Job failed to execute msg = [" + msg + "]");
  }

  auto exited = getJobResponse["exited"].get<bool>();

  if (exited) {
    auto status = getJobResponse["status"].get<std::string>();
    if (status == "cancelled") {
      throw std::runtime_error("Job was cancelled.");
    }
  }

  return exited;
}

std::chrono::microseconds
QCIServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  return std::chrono::microseconds(QCI_CUDAQ_ENDPOINT_POLL_MICRO);
}

// Get the results from a given path
ServerMessage QCIServerHelper::getResults(std::string &resultsGetPath) {
  // Get the headers
  RestHeaders headers = {{"Accept", "*/*"}};

  // Return the results from the client
  return restClient.get(resultsGetPath, "", headers);
}

// Process the results from a job
cudaq::sample_result
QCIServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_DBG("jobId: {}", jobId);

  auto resultsGetPath = postJobResponse.at("resultUrl").get<std::string>();
  // Get the results
  auto results = getResults(resultsGetPath);
  CUDAQ_INFO("results: {}", results.dump());

  if (outputNames.find(jobId) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobId);

  auto const &output_names = outputNames[jobId];
  for (auto const &[result, info] : output_names)
    CUDAQ_INFO("Qubit {} Result {} Name {}", info.qubitNum, result,
               info.registerName);

  auto const &index = results.at("index");
  std::map<std::string, std::vector<std::size_t>> registerMap;
  std::map<std::size_t, std::size_t> globalQubitMap;

  for (auto const &[_, info] : output_names) {
    for (std::size_t i = 0; auto const &entry : index) {
      if (info.registerName == entry[0].get<std::string>() &&
          info.qubitNum == entry[1].get<std::size_t>()) {
        globalQubitMap[info.qubitNum] = i;
        auto &indices = registerMap[info.registerName];
        if (std::find(indices.begin(), indices.end(), i) == indices.end()) {
          indices.push_back(i);
        }
        break;
      }
      ++i;
    }
  }

  std::vector<ExecutionResult> srs;
  std::vector<std::vector<std::size_t>> registerQubitIndicies;
  ExecutionResult ger(GlobalRegisterName);

  for (const auto &[registerName, mis] : registerMap) {
    srs.emplace_back(registerName);
    registerQubitIndicies.emplace_back(mis);
  }

  auto const &allMeasurements = results.at("measurements");
  for (const auto &measurements : allMeasurements) {
    for (std::size_t i = 0; i < registerQubitIndicies.size(); ++i) {
      const auto &qubitIndicies = registerQubitIndicies[i];
      std::string bitstring;
      bitstring.reserve(qubitIndicies.size());

      for (std::size_t idx : qubitIndicies) {
        bitstring += std::to_string(static_cast<int>(measurements[idx]));
      }

      ++srs[i].counts[bitstring];
      srs[i].sequentialData.emplace_back(std::move(bitstring));
    }

    // Global bitstring
    std::string globalBitString;
    globalBitString.reserve(globalQubitMap.size());

    for (const auto &[_, idx] : globalQubitMap) {
      globalBitString += std::to_string(static_cast<int>(measurements[idx]));
    }

    ++ger.counts[globalBitString];
    ger.sequentialData.emplace_back(std::move(globalBitString));
  }

  srs.emplace_back(std::move(ger));

  sample_result sampleResult(srs);

  // reorder according to reorderIdx if needed
  if (auto it = reorderIdx.find(jobId);
      it != reorderIdx.end() && !it->second.empty()) {
    sampleResult.reorder(it->second);
  }

  return sampleResult;
}

std::map<std::string, std::string>
QCIServerHelper::generateRequestHeaders() const {
  std::map<std::string, std::string> headers{
      {"API-Token", backendConfig.at("apiToken")},
      {"Accept", "application/json"},
      {"Authorization", "Bearer " + backendConfig.at("authToken")},
      {"Connection", "keep-alive"},
      {"Content-Type", "application/json"},
      {"User-Agent", "cudaq/" + std::string(cudaq::getVersion())}};

  return headers;
}

RestHeaders QCIServerHelper::getHeaders() { return generateRequestHeaders(); }

} // namespace cudaq

// Register the QCI server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QCIServerHelper, qci)
