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
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <iostream>
#include <thread>

namespace cudaq {

/// @brief Find and set the API and refresh tokens.
void findApiKeyInFile(std::string &apiKey, const std::string &path,
                      std::string &refreshKey);

/// Search for the API key, invokes findApiKeyInFile
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string userSpecifiedConfig = "");

/// @brief The QuantinuumServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Quantinuum REST
/// server.
class QuantinuumServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://nexus.quantinuum.com/";
  /// @brief URL for jobs
  std::string jobUrl = "/api/jobs/v1beta3/";
  /// @brief The machine we are targeting
  std::string machine = "H2-1SC";
  /// @brief Time string, when the last tokens were retrieved
  std::string timeStr = "";
  /// @brief The refresh token
  std::string refreshKey = "";
  /// @brief The API token for the remote server
  std::string apiKey = "";

  std::string userSpecifiedCredentials = "";
  std::string credentialsPath = "";

  /// @brief Quantinuum requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "quantinuum"; }

  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    // Set the machine
    auto iter = backendConfig.find("machine");
    if (iter != backendConfig.end())
      machine = iter->second;

    // Set an alternate base URL if provided
    iter = backendConfig.find("url");
    if (iter != backendConfig.end()) {
      baseUrl = iter->second;
      if (!baseUrl.ends_with("/"))
        baseUrl += "/";
    }

    if (auto project_id = std::getenv("QUANTINUMM_NEXUS_PROJECT_ID"))
      config["project_id"] = project_id;
    else
      config["project_id"] = "";

    iter = backendConfig.find("credentials");
    if (iter != backendConfig.end())
      userSpecifiedCredentials = iter->second;

    parseConfigForCommonParams(config);
  }

  /// @brief Create a job payload for the provided quantum codes
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) override;
};

ServerJobPayload
QuantinuumServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;
  // Construct the job, one per circuit
  for (auto &circuitCode : circuitCodes) {
    /// Ref:
    /// https://nexus.quantinuum.com/api-docs#/jobs/create_job_api_jobs_v1beta3_post
    ServerMessage j;
    j["data"] = ServerMessage::object();
    j["data"]["type"] = "job";
    // Add attributes
    j["data"]["attributes"] = ServerMessage::object();
    j["data"]["attributes"]["name"] = circuitCode.name;
    j["data"]["attributes"]["job_type"] = "execute";
    j["data"]["attributes"]["properties"] = ServerMessage::object();
    // Add definition section
    j["data"]["attributes"]["definition"] = ServerMessage::object();
    j["data"]["attributes"]["definition"]["job_definition_type"] =
        "execute_job_definition";
    j["data"]["attributes"]["definition"]["language"] = "QIR 1.0";
    // Add backend configuration
    j["data"]["attributes"]["definition"]["backend_config"] =
        ServerMessage::object();
    j["data"]["attributes"]["definition"]["backend_config"]["type"] =
        "QuantinuumConfig";
    j["data"]["attributes"]["definition"]["backend_config"]["device_name"] =
        machine;
    // Add program items
    j["data"]["attributes"]["definition"]["items"] = ServerMessage::array();
    ServerMessage item = ServerMessage::object();
    /// ASKME: Should this be autogenerated here?
    // OR use `/api/qir/v1beta/`
    // Ref:
    // https://nexus.quantinuum.com/api-docs#/qir/create_qir_module_api_qir_v1beta_post
    item["program_id"] = "TBD";
    item["n_shots"] = shots;
    j["data"]["attributes"]["definition"]["items"].push_back(item);
    // Add relationships section
    j["data"]["relationships"] = ServerMessage::object();
    j["data"]["relationships"]["project"] = ServerMessage::object();
    j["data"]["relationships"]["project"]["data"] = ServerMessage::object();
    j["data"]["relationships"]["project"]["data"]["id"] =
        backendConfig.at("project_id");
    j["data"]["relationships"]["project"]["data"]["type"] = "project";

    /// ASKME: Field for QIR program?
    j["programs"] = ServerMessage::object();
    j["programs"]["data"] = ServerMessage::object();
    j["programs"]["data"]["type"] = "qir";
    j["programs"]["data"]["content"] = circuitCode.code;

    messages.push_back(j);
    cudaq::info("Payload {}", j.dump(2));
  }

  // Get the headers
  refreshTokens();
  RestHeaders headers = getHeaders();

  CUDAQ_INFO("Created job payload targeting {}", machine);

  // Return the payload with the correct endpoint
  return std::make_tuple(baseUrl + jobUrl, headers, messages);
}

std::string QuantinuumServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["job"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + "job/" + extractJobId(postResponse);
}

std::string QuantinuumServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + "job/" + jobId;
}

bool QuantinuumServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  return false;
}

cudaq::sample_result
QuantinuumServerHelper::processResults(ServerMessage &postJobResponse,
                                       std::string &jobId) {

  return sample_result();
}

RestHeaders QuantinuumServerHelper::getHeaders() {
  std::string accessToken, refreshToken;
  searchAPIKey(accessToken, refreshToken, userSpecifiedCredentials);
  RestHeaders headers;
  headers["Authorization"] = accessToken;
  headers["Content-Type"] = "application/json";
  headers["Connection"] = "keep-alive";
  headers["Accept"] = "*/*";
  return headers;
}

void findApiKeyInFile(std::string &apiKey, const std::string &path,
                      std::string &refreshKey) {
  std::ifstream stream(path);
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());

  std::vector<std::string> lines;
  lines = cudaq::split(contents, '\n');
  for (const std::string &l : lines) {
    std::vector<std::string> keyAndValue = cudaq::split(l, ':');
    if (keyAndValue.size() != 2)
      throw std::runtime_error("Ill-formed configuration file (" + path +
                               "). Key-value pairs must be in `<key> : "
                               "<value>` format. (One per line)");
    cudaq::trim(keyAndValue[0]);
    cudaq::trim(keyAndValue[1]);
    if (keyAndValue[0] == "key")
      apiKey = keyAndValue[1];
    else if (keyAndValue[0] == "refresh")
      refreshKey = keyAndValue[1];
    else
      throw std::runtime_error(
          "Unknown key in configuration file: " + keyAndValue[0] + ".");
  }
  if (apiKey.empty())
    throw std::runtime_error("Empty API key in configuration file (" + path +
                             ").");
  if (refreshKey.empty())
    throw std::runtime_error("Empty refresh key in configuration file (" +
                             path + ").");
}

/// Search for the API key
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string userSpecifiedConfig) {
  std::string hwConfig;
  // Allow someone to tweak this with an environment variable
  if (auto creds = std::getenv("QUANTINUUM_NEXUS_CREDENTIALS"))
    hwConfig = std::string(creds);
  else if (!userSpecifiedConfig.empty())
    hwConfig = userSpecifiedConfig;
  else
    hwConfig = std::string(getenv("HOME")) + std::string("/.quantinuum_config");
  if (cudaq::fileExists(hwConfig)) {
    findApiKeyInFile(key, hwConfig, refreshKey);
  } else {
    throw std::runtime_error(
        "Cannot find Quantinuum Config file with credentials "
        "(~/.quantinuum_config).");
  }

  return hwConfig;
}

/// Refresh the api key and refresh-token
void QuantinuumServerHelper::refreshTokens(bool force_refresh) {}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantinuumServerHelper,
                    quantinuum)
