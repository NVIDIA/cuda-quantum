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

/// @brief The QuantinuumServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Quantinuum REST
/// server.
class QuantinuumServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://nexus.quantinuum.com/";
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
  // Access token lifetime in seconds
  static constexpr int tokenExpirySecs = 5 * 60; // 5 minutes
  // Rest client to send additional requests
  RestClient restClient;

  /// @brief Quantinuum requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Helper to parse the result ID from the job response
  std::string getResultId(ServerMessage &getJobResponse);

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "quantinuum"; }

  RestHeaders getHeaders() override;
  RestCookies getCookies() override;

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

// Load the API key and refresh token from the config file
static void findApiKeyInFile(std::string &apiKey, const std::string &path,
                             std::string &refreshKey, std::string &timeStr) {
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
    else if (keyAndValue[0] == "time")
      timeStr = keyAndValue[1];
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
  // The `time` key is not required.
}

/// Search for the API key, invokes findApiKeyInFile
static std::string searchAPIKey(std::string &key, std::string &refreshKey,
                                std::string &timeStr,
                                std::string userSpecifiedConfig = "") {
  std::string hwConfig;
  // Allow someone to tweak this with an environment variable
  if (auto creds = std::getenv("CUDAQ_QUANTINUUM_CREDENTIALS"))
    hwConfig = std::string(creds);
  else if (!userSpecifiedConfig.empty())
    hwConfig = userSpecifiedConfig;
  else
    hwConfig = std::string(getenv("HOME")) + std::string("/.quantinuum_config");
  if (cudaq::fileExists(hwConfig)) {
    findApiKeyInFile(key, hwConfig, refreshKey, timeStr);
  } else {
    throw std::runtime_error(
        "Cannot find Quantinuum Config file with credentials "
        "(~/.quantinuum_config).");
  }

  return hwConfig;
}

ServerJobPayload
QuantinuumServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Just a placeholder for the job post URL path, headers, and messages
  std::vector<ServerMessage> messages(circuitCodes.size());

  // Get the tokens we need
  credentialsPath =
      searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  refreshTokens();

  // Get the headers
  RestHeaders headers;

  return std::make_tuple(baseUrl + "api/jobs/v1beta", headers, messages);
}

std::string QuantinuumServerHelper::extractJobId(ServerMessage &postResponse) {
  // "job_id": "$response.body#/data.id"
  return postResponse["data"]["id"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + "api/jobs/v1beta3/" + extractJobId(postResponse);
}

std::string QuantinuumServerHelper::constructGetJobPath(std::string &jobId) {
  // TODO: we can use a more lightweight path here.
  // but for now, we will use the overall job path, since we need to get the
  // result Id when it completes.
  return baseUrl + "api/jobs/v1beta3/" + jobId;
}

bool QuantinuumServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Job status strings: "COMPLETED", "QUEUED", "SUBMITTED", "RUNNING",
  // "CANCELLED", "ERROR"
  const std::string jobStatus =
      getJobResponse["data"]["attributes"]["status"]["status"]
          .get<std::string>();
  if (jobStatus == "ERROR") {
    const std::string errorMsg =
        getJobResponse["data"]["attributes"]["status"]["error_detail"]
            .get<std::string>();
    throw std::runtime_error("Job failed with error: " + errorMsg);
  } else if (jobStatus == "CANCELLED") {
    throw std::runtime_error("Job was cancelled.");
  }
  return jobStatus == "COMPLETED";
}

std::string QuantinuumServerHelper::getResultId(ServerMessage &getJobResponse) {
  if (!jobIsDone(getJobResponse)) {
    throw std::runtime_error("Job is not done, cannot retrieve result ID.");
  }
  const auto resultItems =
      getJobResponse["data"]["attributes"]["definition"]["items"];

  // Note: currently, we only support a single result item.
  if (!resultItems.is_array())
    throw std::runtime_error(
        "Expected 'items' to be an array in job response.");
  if (resultItems.size() != 1)
    throw std::runtime_error("Expected exactly one item in 'items' array.");

  const auto &item = resultItems[0];
  if (!item.contains("result_id")) {
    throw std::runtime_error("No 'result_id' found in job response item.");
  }
  return item["result_id"].get<std::string>();
}

cudaq::sample_result
QuantinuumServerHelper::processResults(ServerMessage &jobResponse,
                                       std::string &jobId) {
  const std::string resultId = getResultId(jobResponse);
  const std::string resultPath = baseUrl + "api/results/v1beta3/" + resultId;
  CUDAQ_INFO("Retrieving results from path: {}", resultPath);
  RestHeaders headers = generateRequestHeader();
  RestCookies cookies = getCookies();
  // Retrieve the results
  auto resultResponse = restClient.get(resultPath, "", headers, false, cookies);
  CUDAQ_INFO("Job result response: {}\n", resultResponse.dump());
  auto results = resultResponse["data"]["attributes"]["counts_formatted"];
  CUDAQ_DBG("Count data: {}", results.dump());
  // Get the register names
  auto bitResults = resultResponse["data"]["attributes"]["bits"];
  std::vector<std::string> outputNames;
  for (auto item : bitResults) {
    CUDAQ_DBG("Bit data: {}", item.dump());
    const auto registerName = item[0].get<std::string>();
    outputNames.push_back(registerName);
  }

  std::vector<CountsDictionary> registerResults(outputNames.size());
  cudaq::CountsDictionary globalCounts;
  std::vector<std::string> bitStrings;
  for (const auto &element : results) {
    const auto bitString = element["bitstring"].get<std::string>();
    assert(bitString.length() == outputNames.size());
    const auto count = element["count"].get<std::size_t>();
    globalCounts[bitString] = count;
    for (std::size_t i = 0; i < count; ++i) {
      bitStrings.push_back(bitString);
    }
    // Populate the register results
    for (std::size_t i = 0; i < outputNames.size(); ++i) {
      registerResults[i][std::string{bitString[i]}] += count;
    }
  }
  std::vector<cudaq::ExecutionResult> allResults;
  allResults.reserve(outputNames.size() + 1);
  for (std::size_t i = 0; i < outputNames.size(); ++i) {
    allResults.push_back({registerResults[i], outputNames[i]});
  }

  // Add the global register results
  cudaq::ExecutionResult result{globalCounts, GlobalRegisterName};
  result.sequentialData = bitStrings;
  allResults.push_back(result);
  return cudaq::sample_result{allResults};
}

std::map<std::string, std::string>
QuantinuumServerHelper::generateRequestHeader() const {
  std::map<std::string, std::string> headers{
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

RestHeaders QuantinuumServerHelper::getHeaders() {
  return generateRequestHeader();
}

RestCookies QuantinuumServerHelper::getCookies() {
  if (apiKey.empty() || refreshKey.empty()) {
    searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  }
  if (refreshKey.empty()) {
    throw std::runtime_error(
        "Cannot get cookies, refresh key is empty. Please check your "
        "configuration.");
  }
  refreshTokens();
  return {{"myqos_id", apiKey}};
}

/// Refresh the api key and refresh-token
void QuantinuumServerHelper::refreshTokens(bool force_refresh) {
  if (refreshKey.empty()) {
    throw std::runtime_error(
        "Cannot get refresh access token, refresh key is empty.");
  }
  std::mutex m;
  std::lock_guard<std::mutex> l(m);
  auto now = std::chrono::high_resolution_clock::now();

  // If we are getting close to an 30 min, then we will refresh
  const bool needsRefresh = [&]() {
    // If the time string is empty, we probably need to refresh`
    if (timeStr.empty()) {
      return true;
    }

    // We first check how much time has elapsed since the
    // existing refresh key was created
    std::int64_t timeAsLong = std::stol(timeStr);
    std::chrono::high_resolution_clock::duration d(timeAsLong);
    std::chrono::high_resolution_clock::time_point oldTime(d);
    auto secondsDuration =
        1e-3 *
        std::chrono::duration_cast<std::chrono::milliseconds>(now - oldTime);

    return secondsDuration.count() * (1. / tokenExpirySecs) > .85;
  }();

  if (needsRefresh || force_refresh) {
    cudaq::info("Refreshing id-token");
    RestHeaders cookies{{"myqos_oat", refreshKey}};
    RestCookies headers = generateRequestHeader();
    nlohmann::json j;
    auto response_json =
        restClient.post(baseUrl, "auth/tokens/refresh", j, headers, cookies);
    const auto iter = cookies.find("myqos_id");
    if (iter == cookies.end())
      throw std::runtime_error("Failed to refresh API key, 'myqos_id' not "
                               "found in response cookies.");
    apiKey = iter->second;
    std::ofstream out(credentialsPath);
    out << "key:" << apiKey << '\n';
    out << "refresh:" << refreshKey << '\n';
    out << "time:" << now.time_since_epoch().count() << '\n';
    timeStr = std::to_string(now.time_since_epoch().count());
  }
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantinuumServerHelper,
                    quantinuum)
