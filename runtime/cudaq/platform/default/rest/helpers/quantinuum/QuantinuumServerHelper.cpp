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
  /// @brief URL for jobs
  std::string jobUrl = "api/jobs/v1beta3/";
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
  static constexpr int tokenExpirySecs = 60;
  // Rest client to send additional requests
  RestClient restClient;

  /// @brief Quantinuum requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Create a QIR module from the provided circuit code
  ServerMessage createQIRModule(const KernelExecution &circuitCode);

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

    if (auto project_id = std::getenv("QUANTINUUM_NEXUS_PROJECT_ID"))
      backendConfig["project_id"] = project_id;
    else
      backendConfig["project_id"] = "";

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

ServerMessage
QuantinuumServerHelper::createQIRModule(const KernelExecution &circuitCode) {
  ServerMessage qir;
  /// Ref:
  /// https://nexus.quantinuum.com/api-docs#/qir/create_qir_module_api_qir_v1beta_post
  qir["data"] = ServerMessage::object();
  qir["data"]["type"] = "qir";
  // Add attributes
  qir["data"]["attributes"] = ServerMessage::object();
  qir["data"]["attributes"]["name"] = circuitCode.name;
  qir["data"]["attributes"]["description"] = "Generated by CUDA-Q";
  qir["data"]["attributes"]["properties"] = ServerMessage::object();
  qir["data"]["attributes"]["contents"] = circuitCode.code;
  // Add relationships section
  qir["data"]["relationships"] = ServerMessage::object();
  qir["data"]["relationships"]["project"] = ServerMessage::object();
  qir["data"]["relationships"]["project"]["data"] = ServerMessage::object();
  qir["data"]["relationships"]["project"]["data"]["id"] =
      backendConfig.at("project_id");
  qir["data"]["relationships"]["project"]["data"]["type"] = "project";
  return qir;
}

ServerJobPayload
QuantinuumServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Just a placeholder for the job post URL path, headers, and messages
  std::vector<ServerMessage> messages;

  // Get the tokens we need
  credentialsPath =
      searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  refreshTokens();

  RestHeaders headers = generateRequestHeader();
  RestCookies cookies = getCookies();
  // Construct the job, one per circuit
  for (auto &circuitCode : circuitCodes) {
    // First create a QIR module, and then use its ID in the job
    ServerMessage qir = createQIRModule(circuitCode);
    // Post the QIR module to the server and extract the program ID
    auto response = restClient.post(baseUrl, "api/qir/v1beta/", qir, headers,
                                    true, false, cookies);

    CUDAQ_INFO("QIR creation response: {}", response.dump(2));
    std::string programId = response["data"]["id"].get<std::string>();

    /// Ref:
    /// https://nexus.quantinuum.com/api-docs#/jobs/create_job_api_jobs_v1beta3_post
    ServerMessage j;
    j["data"] = ServerMessage::object();
    j["data"]["type"] = "job";
    // Add attributes
    j["data"]["attributes"] = ServerMessage::object();
    // Construct a unique name for the job, by appending current timestamp
    auto timestamp =
        fmt::format("{:%Y-%m-%d_%H:%M:%S}", std::chrono::system_clock::now());
    j["data"]["attributes"]["name"] =
        fmt::format("{}_{}", circuitCode.name, timestamp);
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
    item["program_id"] = programId;
    item["n_shots"] = shots;
    j["data"]["attributes"]["definition"]["items"].push_back(item);
    // Add relationships section
    j["data"]["relationships"] = ServerMessage::object();
    j["data"]["relationships"]["project"] = ServerMessage::object();
    j["data"]["relationships"]["project"]["data"] = ServerMessage::object();
    j["data"]["relationships"]["project"]["data"]["id"] =
        backendConfig.at("project_id");
    j["data"]["relationships"]["project"]["data"]["type"] = "project";

    messages.push_back(j);
    CUDAQ_DBG("Payload {}", j.dump(2));
  }

  CUDAQ_INFO("Created job payload targeting {}", machine);

  // Return the payload with the correct endpoint
  return cudaq::toServerJobPayload(
      std::make_tuple(baseUrl + jobUrl, headers, messages, cookies));
}

std::string QuantinuumServerHelper::extractJobId(ServerMessage &postResponse) {
  // "job_id": "$response.body#/data.id"
  return postResponse["data"]["id"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + jobUrl + extractJobId(postResponse);
}

std::string QuantinuumServerHelper::constructGetJobPath(std::string &jobId) {
  // TODO: we can use a more lightweight path here.
  // but for now, we will use the overall job path, since we need to get the
  // result Id when it completes.
  return baseUrl + jobUrl + jobId;
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
  auto resultResponse = restClient.get(resultPath, "", headers, false, cookies);
  CUDAQ_INFO("Job result response: {}\n", resultResponse.dump());
  auto results = resultResponse["data"]["attributes"]["counts_formatted"];
  CUDAQ_INFO("Count data: {}", results.dump());

  // TODO: handle register mapping and qubit numbers
  // This is just a very basic implementation that assumes all qubits are
  // measured.
  cudaq::CountsDictionary counts;
  std::vector<std::string> bitStrings;
  for (const auto &element : results) {
    const auto bitString = element["bitstring"].get<std::string>();
    const auto count = element["count"].get<std::size_t>();
    counts[bitString] = count;
    for (std::size_t i = 0; i < count; ++i) {
      bitStrings.push_back(bitString);
    }
  }
  cudaq::ExecutionResult result{counts, GlobalRegisterName};
  result.sequentialData = bitStrings;
  return cudaq::sample_result{result};
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
        restClient.post(baseUrl, "auth/tokens/refresh", j, headers, false,
                        false, cookies, &cookies);
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
