/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
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

namespace {
// Endpoint to submit jobs
constexpr const char *JOBS_ENDPOINT = "cudaq/v1/jobs/";
// Time constants
constexpr std::size_t POLLING_INTERVAL_MICRO = 1000000; // 1 second
// Default values
namespace defaults {
/// Base URL for QCI's service.
constexpr const char *API_URL = "https://aqumen.quantumcircuits.com/";
/// This is not a secret nor a credential. QCI uses these to identify the
/// library or app and the version that is originating each network request to
/// both monitor usage and to disable defective or obsolete releases. Each new
/// release of the SDK should incorporate a new API token.
constexpr const char *API_TOKEN =
    "eyJhbGciOiJFZDI1NTE5IiwidHlwIjoiSldUIn0."
    "eyJhdWQiOiJjbGllbnQiLCJleHAiOjIzODkyNzMxMDUsImlhdCI6MTc1ODU1MzEwNSwiaXNz"
    "IjoiYXF1bWVuIiwianRpIjoiMDE5OTcxZWUtZTI4NS03MGM2LWE3NzMtNzhhMGI4MDRmNTVh"
    "IiwibmJmIjoxNzU4NTUzMTA1LCJzdWIiOiIwMTk5NzFlZS1lMjgwLTdmNjktYjU0Ny05ZTc5"
    "YTc2YTEwNTkifQ."
    "-tthQDI6XuiKPkNJ8sEAKlJthG4hTC2-0mcukejlW82eYZa_u1RBf4J5yQ3Z-2J6O4ZNQvC2"
    "MEIOYzvmZ4-HAg";
constexpr const char *MACHINE = "AquSim";
constexpr const char *METHOD = "simulate";
} // namespace defaults
// Configuration keys
namespace config_keys {
constexpr const char *API_URL = "apiUrl";
constexpr const char *API_TOKEN = "apiToken";
constexpr const char *AUTH_TOKEN = "authToken";
constexpr const char *MACHINE = "machine";
constexpr const char *METHOD = "method";
constexpr const char *NOISY = "noisy";
constexpr const char *RUSR = "repeat_until_shots_requested";
} // namespace config_keys
} // namespace

namespace cudaq {

/// @brief The QCIServerHelper class extends the ServerHelper class to handle
/// interactions with the QCI server for submitting and retrieving quantum
/// computation jobs.
class QCIServerHelper : public ServerHelper, public QirServerHelper {
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
    CUDAQ_INFO("Initializing Quantum Circuits, Inc. backend.");

    auto apiUrl = getEnvVar("QCI_API_URL", defaults::API_URL, false);
    config[config_keys::API_URL] =
        apiUrl.ends_with("/") ? apiUrl : apiUrl + "/";

    config[config_keys::API_TOKEN] =
        getEnvVar("QCI_API_TOKEN", defaults::API_TOKEN, false);

    config[config_keys::MACHINE] =
        getValueOrDefault(config, config_keys::MACHINE, defaults::MACHINE);

    config[config_keys::METHOD] =
        getValueOrDefault(config, config_keys::METHOD, defaults::METHOD);

    CUDAQ_INFO("QCI backend machine: {} with method: {}",
               config[config_keys::MACHINE], config[config_keys::METHOD]);

    config[config_keys::NOISY] =
        getValueOrDefault(config, config_keys::NOISY, "false");

    config[config_keys::RUSR] =
        getValueOrDefault(config, config_keys::RUSR, "false");

    // Authentication token not required in emulation mode
    bool isTokenRequired = [&]() {
      auto it = config.find("emulate");
      return !(it != config.end() && it->second == "true");
    }();

    config[config_keys::AUTH_TOKEN] =
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

  /// @brief Extract QIR output data from the server's response to a job
  std::string extractOutputLog(ServerMessage &postJobResponse,
                               std::string &jobId) override;
};

// Retrieve an environment variable
std::string QCIServerHelper::getEnvVar(const std::string &key,
                                       const std::string &defaultVal,
                                       const bool isRequired) const {
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, either return the default or throw an
  // exception
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error(key + " environment variable is not set.");
    else
      return defaultVal;
  }
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

  // Note that the options can be set as string dictionary from other
  // application(s) which call the CUDA-Q client.
  // Ref: https://github.com/NVIDIA/cuda-quantum/issues/3525
  auto toBool = [](const std::string &value) {
    return value == "True" || value == "true" || value == "1";
  };

  bool rusr = toBool(backendConfig.at(config_keys::RUSR));
  bool noisy = toBool(backendConfig.at(config_keys::NOISY));

  for (auto &circuitCode : circuitCodes) {
    ServerMessage job;
    job["code"] = circuitCode.code;
    job["name"] = circuitCode.name;
    // Target-specific parameters
    job[config_keys::MACHINE] = backendConfig.at(config_keys::MACHINE);
    job[config_keys::METHOD] = backendConfig.at(config_keys::METHOD);
    job["options"] = nlohmann::json::object();
    job["options"]["aqusim"] = {{"shots", shots},
                                {config_keys::NOISY, noisy},
                                {config_keys::RUSR, rusr}};
    if (rusr)
      job["options"]["compiler"] = {{"shots_requested", shots}};
    job["options"]["qpu"] = {{"shots", shots}};

    messages.push_back(job);
  }

  RestHeaders headers = generateRequestHeaders();
  return std::make_tuple(backendConfig.at(config_keys::API_URL) + JOBS_ENDPOINT,
                         headers, messages);
}

std::string QCIServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string QCIServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return backendConfig.at(config_keys::API_URL) + JOBS_ENDPOINT +
         extractJobId(postResponse);
}

std::string QCIServerHelper::constructGetJobPath(std::string &jobId) {
  return backendConfig.at(config_keys::API_URL) + JOBS_ENDPOINT + jobId;
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
  return std::chrono::microseconds(POLLING_INTERVAL_MICRO);
}

std::string QCIServerHelper::getOutputLog(std::string &outputLogPath) {
  RestHeaders headers = {{"Accept", "*/*"}};
  // The path returns TSV text
  return restClient.getRawText(outputLogPath, "", headers);
}

cudaq::sample_result
QCIServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_INFO("jobId: {}", jobId);
  auto outputPath = postJobResponse.at("outputUrl").get<std::string>();
  auto qirResults = getOutputLog(outputPath);
  return createSampleResultFromQirOutput(qirResults);
}

std::map<std::string, std::string>
QCIServerHelper::generateRequestHeaders() const {
  std::map<std::string, std::string> headers{
      {"API-Token", backendConfig.at(config_keys::API_TOKEN)},
      {"Accept", "application/json"},
      {"Authorization", "Bearer " + backendConfig.at(config_keys::AUTH_TOKEN)},
      {"Connection", "keep-alive"},
      {"Content-Type", "application/json"},
      {"User-Agent", "cudaq/" + std::string(cudaq::getVersion())}};

  return headers;
}

RestHeaders QCIServerHelper::getHeaders() { return generateRequestHeaders(); }

std::string QCIServerHelper::extractOutputLog(ServerMessage &postJobResponse,
                                              std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_INFO("jobId: {}", jobId);
  auto outputPath = postJobResponse.at("outputUrl").get<std::string>();
  return getOutputLog(outputPath);
}

} // namespace cudaq

// Register the QCI server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QCIServerHelper, qci)
