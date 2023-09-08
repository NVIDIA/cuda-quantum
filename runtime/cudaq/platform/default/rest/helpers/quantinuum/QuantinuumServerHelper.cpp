/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

/// @brief Find and set the API and refresh tokens, and the time string.
void findApiKeyInFile(std::string &apiKey, const std::string &path,
                      std::string &refreshKey, std::string &timeStr);

/// Search for the API key, invokes findApiKeyInFile
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string &timeStr,
                         std::string userSpecifiedConfig = "");

/// @brief The QuantinuumServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Quantinuum REST
/// server.
class QuantinuumServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://qapi.quantinuum.com/v1/";
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

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

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

    iter = backendConfig.find("credentials");
    if (iter != backendConfig.end())
      userSpecifiedCredentials = iter->second;
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
  for (auto &circuitCode : circuitCodes) {
    // Construct the job itself
    ServerMessage j;
    j["machine"] = machine;
    j["language"] = "QIR 1.0";
    j["program"] = circuitCode.code;
    j["priority"] = "normal";
    j["count"] = shots;
    j["options"] = nullptr;
    j["name"] = circuitCode.name;
    messages.push_back(j);
  }

  // Get the tokens we need
  credentialsPath =
      searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  refreshTokens();

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  cudaq::info(
      "Created job payload for quantinuum, language is QIR 1.0, targeting {}",
      machine);

  // return the payload
  return std::make_tuple(baseUrl + "job", headers, messages);
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
  auto status = getJobResponse["status"].get<std::string>();
  if (status == "failed") {
    std::string msg = "";
    if (getJobResponse.count("error"))
      msg = getJobResponse["error"]["text"].get<std::string>();
    throw std::runtime_error("Job failed to execute msg = [" + msg + "]");
  }

  return status == "completed";
}

cudaq::sample_result
QuantinuumServerHelper::processResults(ServerMessage &postJobResponse,
                                       std::string &jobId) {
  // Results come back as a map of vectors. Each map key corresponds to a qubit
  // and its corresponding vector holds the measurement results in each shot:
  //      { "results" : { "r0" : ["0", "0", ...],
  //                      "r1" : ["1", "0", ...]  } }
  auto results = postJobResponse["results"];

  cudaq::info("Results message: {}", results.dump());

  std::vector<ExecutionResult> srs;

  // Populate individual registers' results into srs
  for (auto &[registerName, result] : results.items()) {
    auto bitResults = result.get<std::vector<std::string>>();
    CountsDictionary thisRegCounts;
    for (auto &b : bitResults)
      thisRegCounts[b]++;
    srs.emplace_back(thisRegCounts, registerName);
    srs.back().sequentialData = bitResults;
  }

  // For each shot, we concatenate the measurements results of all qubits.
  auto begin = results.begin();
  std::vector<std::string> bitstrings =
      begin.value().get<std::vector<std::string>>();
  for (auto it = ++begin, end = results.end(); it != end; ++it) {
    auto bitResults = it.value().get<std::vector<std::string>>();
    for (size_t i = 0; auto &bit : bitResults)
      bitstrings[i++] += bit;
  }

  cudaq::CountsDictionary counts;
  for (auto &b : bitstrings)
    counts[b]++;

  // Store the combined results into the global register
  srs.emplace_back(counts, GlobalRegisterName);
  srs.back().sequentialData = bitstrings;
  return sample_result(srs);
}

std::map<std::string, std::string>
QuantinuumServerHelper::generateRequestHeader() const {
  std::string apiKey, refreshKey, timeStr;
  searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  std::map<std::string, std::string> headers{
      {"Authorization", apiKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

RestHeaders QuantinuumServerHelper::getHeaders() {
  return generateRequestHeader();
}

/// Refresh the api key and refresh-token
void QuantinuumServerHelper::refreshTokens(bool force_refresh) {
  std::mutex m;
  std::lock_guard<std::mutex> l(m);
  RestClient client;
  auto now = std::chrono::high_resolution_clock::now();

  // If the time string is empty, let's add it
  if (timeStr.empty()) {
    timeStr = std::to_string(now.time_since_epoch().count());
    std::ofstream out(credentialsPath);
    out << "key:" << apiKey << '\n';
    out << "refresh:" << refreshKey << '\n';
    out << "time:" << timeStr << '\n';
  }

  // We first check how much time has elapsed since the
  // existing refresh key was created
  std::int64_t timeAsLong = std::stol(timeStr);
  std::chrono::high_resolution_clock::duration d(timeAsLong);
  std::chrono::high_resolution_clock::time_point oldTime(d);
  auto secondsDuration =
      1e-3 *
      std::chrono::duration_cast<std::chrono::milliseconds>(now - oldTime);

  // If we are getting close to an 30 min, then we will refresh
  bool needsRefresh = secondsDuration.count() * (1. / 1800.) > .85;
  if (needsRefresh || force_refresh) {
    cudaq::info("Refreshing id-token");
    std::stringstream ss;
    ss << "\"refresh-token\":\"" << refreshKey << "\"";
    auto headers = generateRequestHeader();
    nlohmann::json j;
    j["refresh-token"] = refreshKey;
    auto response_json = client.post(baseUrl, "login", j, headers);
    apiKey = response_json["id-token"].get<std::string>();
    refreshKey = response_json["refresh-token"].get<std::string>();
    std::ofstream out(credentialsPath);
    out << "key:" << apiKey << '\n';
    out << "refresh:" << refreshKey << '\n';
    out << "time:" << now.time_since_epoch().count() << '\n';
    timeStr = std::to_string(now.time_since_epoch().count());
  }
}

void findApiKeyInFile(std::string &apiKey, const std::string &path,
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

/// Search for the API key
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string &timeStr,
                         std::string userSpecifiedConfig) {
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

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantinuumServerHelper,
                    quantinuum)
