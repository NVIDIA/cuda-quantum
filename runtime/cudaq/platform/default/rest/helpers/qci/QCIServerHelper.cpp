/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates and Contributors. *
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
class QCIServerHelper : public ServerHelper, public QirServerHelper {
private:
  /// @brief Default API token. This is not a secret nor a credential. QCI uses
  /// these to identify the library or app and the version that is originating
  /// each network request to both monitor usage and to disable defective or
  /// obsolete releases. Each new release of the SDK should incorporate a new
  /// API token.
  const std::string DEFAULT_API_TOKEN =
      "eyJhbGciOiJFZDI1NTE5IiwidHlwIjoiSldUIn0."
      "eyJhdWQiOiJjbGllbnQiLCJleHAiOjIzODkyNzMxMDUsImlhdCI6MTc1ODU1MzEwNSwiaXNz"
      "IjoiYXF1bWVuIiwianRpIjoiMDE5OTcxZWUtZTI4NS03MGM2LWE3NzMtNzhhMGI4MDRmNTVh"
      "IiwibmJmIjoxNzU4NTUzMTA1LCJzdWIiOiIwMTk5NzFlZS1lMjgwLTdmNjktYjU0Ny05ZTc5"
      "YTc2YTEwNTkifQ."
      "-tthQDI6XuiKPkNJ8sEAKlJthG4hTC2-0mcukejlW82eYZa_u1RBf4J5yQ3Z-2J6O4ZNQvC2"
      "MEIOYzvmZ4-HAg";

  /// @brief Default base URL for QCI's service.
  const std::string DEFAULT_API_URL = "https://aqumen.quantumcircuits.com/";

  /// @brief Default machine, the simulator.
  const std::string DEFAULT_MACHINE = "AquSim";

  /// @brief Default action to perform
  const std::string DEFAULT_METHOD = "simulate";

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

    config["method"] = getValueOrDefault(config, "method", DEFAULT_METHOD);
    CUDAQ_INFO("QCI backend machine: {} with method: {}", config["machine"],
               config["method"]);

    config["noisy"] = getValueOrDefault(config, "noisy", "false");
    config["rusr"] =
        getValueOrDefault(config, "repeat_until_shots_requested", "false");

    // Authentication token not required in emulation mode
    bool isTokenRequired = [&]() {
      auto it = config.find("emulate");
      return !(it != config.end() && it->second == "true");
    }();

    config["authToken"] =
        getEnvVar("QCI_AUTH_TOKEN", "QCI_AUTH_TOKEN_NOT_SET", isTokenRequired);

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

  /// @brief Retrieve the QIR output log from the provided path.
  std::string getOutputLog(std::string &outputLogPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;

  // Extract QIR output data
  std::string extractOutputLog(ServerMessage &postJobResponse,
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

  bool rusr = backendConfig.at("rusr") == "true";
  bool noisy = backendConfig.at("noisy") == "true";

  for (auto &circuitCode : circuitCodes) {
    ServerMessage job;
    job["code"] = circuitCode.code;
    job["name"] = circuitCode.name;
    // Target-specific parameters
    job["machine"] = backendConfig.at("machine");
    job["method"] = backendConfig.at("method");
    job["options"] = nlohmann::json::object();
    job["options"]["compiler"] = {{"shots", shots}, {"rusr", rusr}};
    job["options"]["aqusim"] = {{"noisy", noisy}, {"rusr", rusr}};
    job["options"]["qpu"] = {{"rusr", true}};

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

// Get the QIR output log from a given path
std::string QCIServerHelper::getOutputLog(std::string &outputLogPath) {
  RestHeaders headers = {{"Accept", "*/*"}};
  // The path returns TSV text
  return restClient.getRawText(outputLogPath, "", headers);
}

// Process the results from a job
cudaq::sample_result
QCIServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_DBG("jobId: {}", jobId);
  auto outputPath = postJobResponse.at("outputUrl").get<std::string>();
  auto qirResults = getOutputLog(outputPath);
  return createSampleResultFromQirOutput(qirResults);
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

// Extract QIR output data
std::string QCIServerHelper::extractOutputLog(ServerMessage &postJobResponse,
                                              std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_DBG("jobId: {}", jobId);
  auto outputPath = postJobResponse.at("outputUrl").get<std::string>();
  return getOutputLog(outputPath);
}

} // namespace cudaq

// Register the QCI server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QCIServerHelper, qci)
