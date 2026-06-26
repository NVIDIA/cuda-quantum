/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/RestClient.h"
#include "common/ResultReconstruction.h"
#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <regex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

using nlohmann::json;

namespace cudaq {

/// @brief Find and set the API and refresh tokens, and the time string.
void findApiKeyInFile(std::string &apiKey, const std::string &path,
                      std::string &refreshKey, std::string &timeStr,
                      std::string &credentials);

/// Search for the API key, invokes findApiKeyInFile
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string &credentials, std::string &timeStr,
                         std::string userSpecifiedConfig = "");

static bool isAnyonResultBit(char bit) { return bit == '0' || bit == '1'; }

/// @brief The implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote REST
/// server.
class AnyonServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://api.anyon.cloud:5000/";
  /// @brief The machine we are targeting.
  std::string machine = "telegraph-8q"; //"berkeley-25q";//
  /// @brief Time string, when the last tokens were retrieved
  std::string timeStr = "";
  /// @brief The refresh token
  std::string refreshKey = "";
  /// @brief The API token for the remote server
  std::string apiKey = "";
  std::string credentials = "";

  std::string userSpecifiedCredentials = "";
  std::string credentialsPath = "";

  /// @brief The ServerHelper requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;
  RestHeaders generateRequestHeader(std::string) const;

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "anyon"; }
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

  /// @brief Update `passPipeline` with architecture-specific pass options
  void updatePassPipeline(const std::filesystem::path &platformPath,
                          std::string &passPipeline) override;
};

ServerJobPayload
AnyonServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {

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
  credentialsPath = searchAPIKey(apiKey, refreshKey, credentials, timeStr,
                                 userSpecifiedCredentials);
  refreshTokens();

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  CUDAQ_INFO("Created job payload for anyon, language is QIR 1.0, targeting {}",
             machine);

  // return the payload
  return std::make_tuple(baseUrl + "job", headers, messages);
}

std::string AnyonServerHelper::extractJobId(ServerMessage &postResponse) {
  // printf("Extracting ID\n");
  std::string jobToken =
      postResponse[0]["job_token"]
          .get<std::string>(); // The post response is an array [json_data,
                               // http_status_code]
  // printf("Extracted ID %s\n",jobToken.c_str());
  return jobToken;
}

std::string
AnyonServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + "job/" + extractJobId(postResponse);
}

std::string AnyonServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + "job/" + jobId;
}

bool AnyonServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  auto status = getJobResponse[0]["status"]
                    .get<std::string>(); // All job get and post responses at an
                                         // array of [resdata, httpstatuscode]
  if (status == "failed") {
    std::string msg = "";
    if (getJobResponse[0].count("error"))
      msg = getJobResponse[0]["error"]["text"].get<std::string>();
    throw std::runtime_error("Job failed to execute msg = [" + msg + "]");
  } else if (status == "waiting") {
    return false;
  } else if (status == "executing") {
    return false;
  } else
    return status == "done";
}

cudaq::sample_result
AnyonServerHelper::processResults(ServerMessage &postJobResponse,
                                  std::string &jobId) {
  // Results come back as a map of vectors. Each map key corresponds to a qubit
  // and its corresponding vector holds the measurement results in each shot:
  //      { "results" : { "r0" : ["0", "0", ...],
  //                      "r1" : ["1", "0", ...]  } }
  auto results = postJobResponse[0]["results"];

  CUDAQ_INFO("Results message: {}", results.dump());

  // The local mock server tests do not support the full named QIR output
  // recording functions. They return a single register keyed by
  // MOCK_SERVER_RESULTS holding the raw concatenated bitstring per shot, with
  // no per-register decomposition. Pass that bitstring through unchanged.
  if (results.begin().key() == "MOCK_SERVER_RESULTS") {
    auto bitResults =
        results.at(results.begin().key()).get<std::vector<std::string>>();
    CountsDictionary counts;
    for (const auto &bits : bitResults)
      counts[bits]++;
    ExecutionResult globalResult{counts, GlobalRegisterName};
    globalResult.sequentialData = bitResults;
    return sample_result({globalResult});
  }

  if (outputNames.find(jobId) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobId);

  auto &output_names = outputNames[jobId];
  for (auto &[result, info] : output_names) {
    CUDAQ_INFO("Qubit {} Result {} Name {}", info.qubitNum, result,
               info.registerName);
    if (!results.contains(info.registerName))
      throw std::runtime_error("Expected to see " + info.registerName +
                               " in the results, but did not see it.");
  }

  // Anyon returns measurement results keyed by register name, each a vector of
  // per-shot bit strings. A register holding several results carries one bit
  // per result in output-position order. Assemble a per-shot flat bitstring
  // indexed by qubit number, then reconstruct the user-visible order and named
  // registers from the enriched output_names through the single bit-index path.
  std::unordered_map<std::string, std::vector<std::size_t>> resultsByRegister;
  for (const auto &[result, info] : output_names)
    resultsByRegister[info.registerName].push_back(result);
  for (auto &[registerName, resultIds] : resultsByRegister)
    std::sort(resultIds.begin(), resultIds.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                return output_names[lhs].outputPosition <
                       output_names[rhs].outputPosition;
              });

  std::size_t flatWidth = 0;
  for (const auto &[result, info] : output_names)
    flatWidth = std::max(flatWidth, info.qubitNum + 1);

  std::size_t nShots =
      results.begin().value().get<std::vector<std::string>>().size();
  std::vector<std::string> flatBitstrings(nShots, std::string(flatWidth, '0'));

  for (const auto &[registerName, resultIds] : resultsByRegister) {
    auto bitResults = results.at(registerName).get<std::vector<std::string>>();
    if (bitResults.size() != nShots)
      throw std::runtime_error("Anyon result register " + registerName +
                               " has an unexpected shot count.");
    for (std::size_t shot = 0; shot < nShots; ++shot) {
      const auto &shotBits = bitResults[shot];
      if (shotBits.size() != 1 && shotBits.size() != resultIds.size())
        throw std::runtime_error("Anyon result register " + registerName +
                                 " contained a measurement with unexpected "
                                 "width.");
      for (std::size_t localPosition = 0; localPosition < resultIds.size();
           ++localPosition) {
        auto bit = shotBits.size() == 1 ? shotBits[0] : shotBits[localPosition];
        if (!isAnyonResultBit(bit))
          throw std::runtime_error("Anyon result register " + registerName +
                                   " contained a non-bit measurement.");
        flatBitstrings[shot][output_names[resultIds[localPosition]].qubitNum] =
            bit;
      }
    }
  }

  // The enriched output_names result map preserves per-shot ordering, so the
  // flat-bitstring shot path keeps Anyon's sequential data intact.
  if (auto resultMap = resultMapForJob(jobId))
    return reconstructSampleResultFromDeviceIndexedBitstringShots(
        flatBitstrings, *resultMap);

  CountsDictionary counts;
  for (const auto &bits : flatBitstrings)
    counts[bits]++;
  ExecutionResult globalResult{counts, GlobalRegisterName};
  globalResult.sequentialData = flatBitstrings;
  return sample_result({globalResult});
}

std::map<std::string, std::string>
AnyonServerHelper::generateRequestHeader() const {
  std::string apiKey, refreshKey, credentials, timeStr;
  searchAPIKey(apiKey, refreshKey, credentials, timeStr,
               userSpecifiedCredentials);
  std::map<std::string, std::string> headers{
      {"Authorization", apiKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

std::map<std::string, std::string>
AnyonServerHelper::generateRequestHeader(std::string authKey) const {
  std::map<std::string, std::string> headers{
      {"Authorization", authKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

RestHeaders AnyonServerHelper::getHeaders() { return generateRequestHeader(); }

/// Refresh the api key and refresh-token
void AnyonServerHelper::refreshTokens(bool force_refresh) {
  std::mutex m;
  std::lock_guard<std::mutex> l(m);
  RestClient client;
  auto now = std::chrono::high_resolution_clock::now();

  if (apiKey.empty()) {
    force_refresh = true;
    if (refreshKey.empty())
      refreshKey = credentials;
  }
  if (timeStr.empty()) {
    timeStr = std::to_string(now.time_since_epoch().count());
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
    CUDAQ_INFO("Refreshing id_token");
    std::stringstream ss;
    ss << "\"refresh_token\":\"" << refreshKey << "\"";
    auto headers = generateRequestHeader(refreshKey);
    nlohmann::json j;
    j["refresh_token"] = refreshKey;
    auto response_json = client.post(baseUrl, "login", j, headers);
    apiKey = response_json["id_token"].get<std::string>();
    refreshKey = response_json["refresh_token"].get<std::string>();
    std::ofstream out(credentialsPath);
    out << "key:" << apiKey << '\n';
    out << "refresh:" << refreshKey << '\n';
    out << "time:" << now.time_since_epoch().count() << '\n';
    timeStr = std::to_string(now.time_since_epoch().count());
  }
  // If the time string is empty, let's add it
  if (timeStr.empty()) {
    timeStr = std::to_string(now.time_since_epoch().count());
    std::ofstream out(credentialsPath);
    out << "key:" << apiKey << '\n';
    out << "refresh:" << refreshKey << '\n';
    out << "time:" << timeStr << '\n';
  }
}

void findApiKeyInFile(std::string &apiKey, const std::string &path,
                      std::string &refreshKey, std::string &timeStr,
                      std::string &credentials) {
  std::ifstream stream(path);
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());

  std::vector<std::string> lines;
  lines = cudaq::split(contents, '\n');
  nlohmann::json jsoncreds;
  for (const std::string &l : lines) {
    std::vector<std::string> keyAndValue = cudaq::split(l, ':');
    if ((keyAndValue.size() != 2) &&
        ((keyAndValue[0] != "credentials") || (keyAndValue.size() != 4)))
      throw std::runtime_error("Ill-formed configuration file (" + path +
                               "). Key-value pairs must be in `<key> : "
                               "<value>` or `<key> : {username:<username>, "
                               "password:<password>}` format. (One per line)");
    cudaq::trim(keyAndValue[0]);
    cudaq::trim(keyAndValue[1]);
    if (keyAndValue[0] == "key")
      apiKey = keyAndValue[1];
    else if (keyAndValue[0] == "refresh")
      refreshKey = keyAndValue[1];
    else if (keyAndValue[0] == "time")
      timeStr = keyAndValue[1];
    else if (keyAndValue[0] ==
             "credentials") { // If the config file doesn't contain key and
                              // refresh token, we will add the username
                              // password to apikey for BasicHttpAuthentication
                              // and generation of tokens
      std::string linecontent =
          keyAndValue[1] + ":" + keyAndValue[2] + ":" + keyAndValue[3];
      // printf("The credentials read from the .config file is: %s",
      // linecontent.c_str());
      jsoncreds = json::parse(linecontent);
      std::string delim(":");
      std::string username = jsoncreds.at("username");
      std::string passwd = jsoncreds.at("password");
      std::string authInfo = username + delim + passwd;
      // authInfo = base64::to_base64(authInfo);
      authInfo = llvm::encodeBase64(authInfo);
      credentials = "Basic " + authInfo;
    } else
      throw std::runtime_error(
          "Unknown key in configuration file: " + keyAndValue[0] + ".");
  }

  if (credentials.empty() && refreshKey.empty())
    throw std::runtime_error("Empty credentials in configuration file (" +
                             path + ").");
  // The `time` key is not required.
}

/// Search for the API key
std::string searchAPIKey(std::string &key, std::string &refreshKey,
                         std::string &credentials, std::string &timeStr,
                         std::string userSpecifiedConfig) {
  std::string hwConfig;
  // Allow someone to tweak this with an environment variable
  if (auto creds = std::getenv("CUDAQ_ANYON_CREDENTIALS"))
    hwConfig = std::string(creds);
  else if (!userSpecifiedConfig.empty())
    hwConfig = userSpecifiedConfig;
  else
    hwConfig = std::string(getenv("HOME")) + std::string("/.anyon_config");
  if (cudaq::fileExists(hwConfig)) {
    findApiKeyInFile(key, hwConfig, refreshKey, timeStr, credentials);
  } else {
    throw std::runtime_error("Cannot find Anyon Config file with credentials "
                             "(~/.anyon_config).");
  }

  return hwConfig;
}

void AnyonServerHelper::updatePassPipeline(
    const std::filesystem::path &platformPath, std::string &passPipeline) {
  std::string qgate_type = "cgate";
  if (machine.starts_with("berkeley")) {
    qgate_type = "pgate";
  } else if (machine.starts_with("telegraph")) {
    qgate_type = "cgate";
  } else {
    printf("Unidentified machine type %s\n", machine.c_str());
  }
  passPipeline =
      std::regex_replace(passPipeline, std::regex("%Q_GATE%"), qgate_type);

  std::string pathToFile = platformPath / std::string("mapping/anyon") /
                           (machine + std::string(".txt"));
  passPipeline =
      std::regex_replace(passPipeline, std::regex("%QPU_ARCH%"), pathToFile);
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::AnyonServerHelper, anyon)
