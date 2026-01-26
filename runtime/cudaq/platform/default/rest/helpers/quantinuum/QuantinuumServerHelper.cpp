/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuantinuumHelper.h"
#include "common/ExtraPayloadProvider.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <fstream>
#include <iostream>
#include <regex>
#include <thread>

namespace {
/// API endpoints
constexpr const char *authEndpoint = "auth/tokens/refresh";
constexpr const char *projectsEndpoint = "api/projects/v1beta2";
constexpr const char *jobsEndpoint = "api/jobs/v1beta3/";
constexpr const char *qirEndpoint = "api/qir/v1beta/";
// Legacy result endpoint (PYTKET)
constexpr const char *resultsEndpoint = "api/results/v1beta3/";
// NG device result endpoint (QSYS)
constexpr const char *qsysResultsEndpoint = "api/qsys_results/v1beta/";
// Decoder config endpoint
constexpr const char *gpuDecoderConfigEndpoint =
    "api/gpu_decoder_configs/v1beta";
} // namespace

namespace cudaq {

/// @brief The QuantinuumServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Quantinuum REST
/// server.
class QuantinuumServerHelper : public ServerHelper, public QirServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://nexus.quantinuum.com/";
  /// @brief The machine we are targeting
  std::string machine = "H2-1SC";
  /// @brief Max HQC cost
  std::optional<int> maxCost;
  /// @brief Maximum number of qubits
  std::optional<int> maxQubits;
  /// @brief Enable/disable noisy simulation on emulator.
  std::optional<bool> noisySim;
  /// @brief The type of simulator to use if machine is a simulator.
  std::string simulator;
  /// @brief The Nexus project ID
  std::string projectId = "";
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

  /// @brief Retrieve project ID from the project name
  void setProjectId(const std::string &userInput);

  /// @brief Different result type that the service may return
  enum class ResultType { PYTKET, QSYS };
  /// @brief Enum to specify results in a specific format
  enum class QsysResultVersion : int { DEFAULT = 3, RAW = 4 };
  /// @brief Create a server request to create an extra resource on the server.
  ServerMessage createExtraResource(const std::string &type,
                                    const std::string &name,
                                    const std::string &contents);

  /// @brief Return a payload provider if any was configured for this target.
  /// @return Extra payload provider if configured, nullptr otherwise.
  // For example, via the nvq++ CLI or Python set_target, an extra payload
  // provider can be specified. The server helper, in accordance with the
  // service provider API, will handle the integration of this extra payload
  // into the job submission process.
  cudaq::ExtraPayloadProvider *getExtraPayloadProvider();

  /// @brief Helper to parse the result ID from the job response
  std::pair<ResultType, std::string> getResultId(ServerMessage &getJobResponse);
  // Extract QIR output data
  std::string extractOutputLog(ServerMessage &postJobResponse,
                               std::string &jobId) override;

  /// @brief Helper to determine if a completed job returns a result.
  // Some jobs, such as syntax checker jobs, may complete without returning a
  // result.
  bool jobReturnsResult(ServerMessage &postJobResponse) const;

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

    // Set max cost
    iter = backendConfig.find("max_cost");
    if (iter != backendConfig.end()) {
      maxCost = std::stoi(iter->second);
      if (maxCost.value() < 1)
        throw std::runtime_error("max_cost must be a positive integer.");
    }

    // Set max qubits
    iter = backendConfig.find("max_qubits");
    if (iter != backendConfig.end()) {
      maxQubits = std::stoi(iter->second);
      if (maxQubits.value() < 1)
        throw std::runtime_error("max_qubits must be a positive integer.");
    }

    // Noisy simulation
    iter = backendConfig.find("noisy_simulation");
    if (iter != backendConfig.end()) {
      if (iter->second != "true" && iter->second != "false")
        throw std::runtime_error("noisy_simulation must be true or false.");
      noisySim = (iter->second == "true");
    }

    // Simulator name
    iter = backendConfig.find("simulator");
    if (iter != backendConfig.end())
      simulator = iter->second;

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

    // Set project ID
    iter = backendConfig.find("project");
    if (iter != backendConfig.end())
      setProjectId(iter->second);
    else {
      // Emulation does not require a project ID
      iter = backendConfig.find("emulate");
      // if not emulate then throw an error
      if (iter != backendConfig.end() && iter->second == "false")
        throw std::runtime_error("Missing mandatory field for Nexus project. "
                                 "Please provide a valid project name or ID.");
    }

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

void QuantinuumServerHelper::setProjectId(const std::string &userInput) {
  // Get the tokens we need
  credentialsPath =
      searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  refreshTokens();
  RestHeaders headers = generateRequestHeader();
  RestCookies cookies = getCookies();
  // Lambda to validate UUID format. This regex checks for the standard UUID
  // format: 8-4-4-4-12 hexadecimal characters as specified in RFC 4122.
  auto isValidUUID = [](const std::string &inputStr) -> bool {
    // Regular expression for UUID validation
    const std::regex uuidRegex("^[a-fA-F0-9]{8}-"
                               "[a-fA-F0-9]{4}-"
                               "[1-5][a-fA-F0-9]{3}-"
                               "[89abAB][a-fA-F0-9]{3}-"
                               "[a-fA-F0-9]{12}$");
    return std::regex_match(inputStr, uuidRegex);
  };
  // If the user input is a UUID, check if it refers to valid Nexus project
  if (isValidUUID(userInput)) {
    /// Ref:
    /// https://nexus.quantinuum.com/api-docs#/projects/get_project_api_projects_v1beta2__project_id__get
    auto response =
        restClient.get(baseUrl, std::string(projectsEndpoint) + '/' + userInput,
                       headers, false, cookies);
    if (response.contains("data") && response["data"].contains("id") &&
        response["data"]["id"].is_string()) {
      projectId = response["data"]["id"].get<std::string>();
      return;
    }
  }
  // If not, we need to search for the project by name
  /// Ref:
  /// https://nexus.quantinuum.com/api-docs#/projects/list_projects_api_projects_v1beta2_get
  std::string filter = "?filter%5Bname%5D=" + userInput;
  auto response = restClient.get(baseUrl, projectsEndpoint + filter, headers,
                                 false, cookies);
  if (response.contains("data") && response["data"].is_array() &&
      response["data"].size() > 0 && response["data"][0].contains("id") &&
      response["data"][0]["id"].is_string())
    projectId = response["data"][0]["id"].get<std::string>();
  else
    throw std::runtime_error(
        "Project not found. Please provide valid Nexus project name or ID.");
}

ServerMessage
QuantinuumServerHelper::createExtraResource(const std::string &type,
                                            const std::string &name,
                                            const std::string &contents) {
  ServerMessage resource;
  resource["data"] = ServerMessage::object();
  resource["data"]["type"] = type;
  // Add attributes
  resource["data"]["attributes"] = ServerMessage::object();
  resource["data"]["attributes"]["name"] = name;
  resource["data"]["attributes"]["description"] = "Generated by CUDA-Q";
  resource["data"]["attributes"]["properties"] = ServerMessage::object();
  resource["data"]["attributes"]["contents"] = contents;
  // Add relationships section
  resource["data"]["relationships"] = ServerMessage::object();
  resource["data"]["relationships"]["project"] = ServerMessage::object();
  resource["data"]["relationships"]["project"]["data"] =
      ServerMessage::object();
  resource["data"]["relationships"]["project"]["data"]["id"] = projectId;
  resource["data"]["relationships"]["project"]["data"]["type"] = "project";
  return resource;
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

  // Any additional resources needed for the job
  auto *extraPayloadProvider = getExtraPayloadProvider();
  // GPU decoder config UUID, if any.
  std::string gpuDecoderConfigId;
  if (extraPayloadProvider) {
    // Use the extra payload provider to modify the job payload
    if (extraPayloadProvider->getPayloadType() != "gpu_decoder_config") {
      // Currently, only `gpu_decoder_config` extra payloads are supported.
      throw std::runtime_error(
          fmt::format("Invalid extra payload provider type '{}'. This is not "
                      "supported on this target.",
                      extraPayloadProvider->getPayloadType()));
    }
    const auto decoderConfigYmlStr =
        extraPayloadProvider->getExtraPayload(runtimeTarget);
    CUDAQ_DBG("[Decoder Config] Received the YML config:\n{}",
              decoderConfigYmlStr);

    const std::string resourceType = extraPayloadProvider->getPayloadType();
    // Create a time-stamped name for the payload
    const auto timestamp =
        fmt::format("{:%Y-%m-%d_%H:%M:%S}", std::chrono::system_clock::now());
    const std::string resourceName =
        fmt::format("{}_{}", "cudaq_decoder_config", timestamp);

    const std::string resourceContent = llvm::encodeBase64(decoderConfigYmlStr);
    // Post the resource to the server and extract the handle reference
    ServerMessage resourceUploadMessage =
        createExtraResource(resourceType, resourceName, resourceContent);

    auto response = restClient.post(baseUrl, gpuDecoderConfigEndpoint,
                                    resourceUploadMessage, headers, true, false,
                                    getCookies());
    if (!response.contains("data") || !response["data"].contains("id") ||
        !response["data"]["id"].is_string())
      throw std::runtime_error("Failed to upload resource: " + resourceName +
                               ". Response: " + response.dump(2));

    gpuDecoderConfigId = response["data"]["id"].get<std::string>();
    CUDAQ_INFO("[Decoder Config] Uploaded GPU decoder config resource "
               "successfully with ID: {}",
               gpuDecoderConfigId);
  }

  // Construct the job, one per circuit
  for (auto &circuitCode : circuitCodes) {
    // First create a QIR module, and then use its ID in the job
    ServerMessage qir =
        createExtraResource("qir", circuitCode.name, circuitCode.code);
    // Post the QIR module to the server and extract the program ID
    auto response = restClient.post(baseUrl, qirEndpoint, qir, headers, true,
                                    false, cookies);
    if (!response.contains("data") || !response["data"].contains("id") ||
        !response["data"]["id"].is_string())
      throw std::runtime_error(
          "Failed to create QIR module for circuit: " + circuitCode.name +
          ". Response: " + response.dump(2));
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
    // On Helios devices, we need to specify max-cost and max-qubits unless it's
    // a syntax checker
    if (machine.starts_with("Helios") && !machine.ends_with("SC")) {
      std::vector<std::string> errors;
      if (!maxCost.has_value())
        errors.push_back("Please specify maximum HQC cost "
                         "(`--quantinuum-max-cost <val>` when compiling with "
                         "nvq++ or `max_cost=<val>` in Python `set_target`)");
      if (!maxQubits.has_value())
        errors.push_back(
            "Please specify maximum number of qubits (`--quantinuum-max-qubits "
            "<val>` when compiling with nvq++ or `max_qubits=<val>` in Python "
            "`set_target`)");
      if (!errors.empty())
        throw std::runtime_error(
            fmt::format("Missing required configuration for device '{}': {}",
                        machine, fmt::join(errors, "; ")));
    }

    if (maxCost.has_value())
      j["data"]["attributes"]["definition"]["backend_config"]["max_cost"] =
          maxCost.value();

    if (maxQubits.has_value()) {
      j["data"]["attributes"]["definition"]["backend_config"]
       ["compiler_options"] = ServerMessage::object();
      j["data"]["attributes"]["definition"]["backend_config"]
       ["compiler_options"]["max-qubits"] = maxQubits.value();
    }

    if (noisySim.has_value() && machine.ends_with("E"))
      j["data"]["attributes"]["definition"]["backend_config"]
       ["noisy_simulation"] = noisySim.value() ? "true" : "false";

    if (!simulator.empty())
      j["data"]["attributes"]["definition"]["backend_config"]["simulator"] =
          simulator;

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
    j["data"]["relationships"]["project"]["data"]["id"] = projectId;
    j["data"]["relationships"]["project"]["data"]["type"] = "project";
    // There is a GPU decoder config resource associated with this job
    if (!gpuDecoderConfigId.empty()) {
      j["data"]["attributes"]["definition"]["gpu_decoder_config_id"] =
          gpuDecoderConfigId;
    }
    messages.push_back(j);
  }
  CUDAQ_INFO("Created job payload targeting {}", machine);
  // Return the payload with the correct endpoint
  return std::make_tuple(baseUrl + jobsEndpoint, headers, messages);
}

std::string QuantinuumServerHelper::extractJobId(ServerMessage &postResponse) {
  // "job_id": "$response.body#/data.id"
  return postResponse["data"]["id"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + jobsEndpoint + extractJobId(postResponse);
}

std::string QuantinuumServerHelper::constructGetJobPath(std::string &jobId) {
  // TODO: we can use a more lightweight path here.
  // but for now, we will use the overall job path, since we need to get the
  // result Id when it completes.
  return baseUrl + jobsEndpoint + jobId;
}

bool QuantinuumServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Job status strings: "COMPLETED", "QUEUED", "SUBMITTED", "RUNNING",
  // "CANCELLED", "ERROR", "CANCELLING", "RETRYING", "TERMINATED", "DEPLETED"
  const std::string jobStatus =
      getJobResponse["data"]["attributes"]["status"]["status"]
          .get<std::string>();
  // Handle error conditions:
  if (jobStatus == "ERROR") {
    const std::string errorMsg =
        getJobResponse["data"]["attributes"]["status"]["error_detail"]
            .get<std::string>();
    throw std::runtime_error("Job failed with error: " + errorMsg);
  } else if (jobStatus == "CANCELLED") {
    // Note: if the status is "CANCELLING", we will let it resolve to CANCELLED
    // before throwing.
    throw std::runtime_error("Job was cancelled.");
  } else if (jobStatus == "TERMINATED") {
    throw std::runtime_error("Job was terminated.");
  } else if (jobStatus == "DEPLETED") {
    throw std::runtime_error("Job failed due to depleted credits. Please check "
                             "your max-cost setting or the account credits.");
  }

  if (jobStatus == "COMPLETED") {
    if (!jobReturnsResult(getJobResponse))
      return true;
    // Check if the response contains the result ID
    // In some cases, the status may be "COMPLETED" but the result ID
    // is not yet available, so we will check for that.
    return getResultId(getJobResponse).second != "";
  }
  // Other status codes, e.g., QUEUED/SUBMITTED/CANCELLING/RETRYING/RUNNING,
  // mean the job is not done yet.
  return false;
}

std::pair<QuantinuumServerHelper::ResultType, std::string>
QuantinuumServerHelper::getResultId(ServerMessage &getJobResponse) {
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
    return std::make_pair(QuantinuumServerHelper::ResultType::PYTKET,
                          ""); // No result ID available yet
  }

  const std::string resultTypeStr = item["result_type"].get<std::string>();
  const std::string resultId = item["result_id"].get<std::string>();
  if (resultTypeStr == "QSYS") {
    // This is a QSYS result
    return std::make_pair(QuantinuumServerHelper::ResultType::QSYS, resultId);
  } else if (resultTypeStr == "PYTKET") {
    // This is a PYTKET result
    return std::make_pair(QuantinuumServerHelper::ResultType::PYTKET, resultId);
  } else {
    throw std::runtime_error("Unknown result type: " + resultTypeStr);
  }
}

cudaq::sample_result
QuantinuumServerHelper::processResults(ServerMessage &jobResponse,
                                       std::string &jobId) {
  const auto [resultType, resultId] = getResultId(jobResponse);
  if (resultId.empty()) {
    if (!jobReturnsResult(jobResponse))
      return cudaq::sample_result(cudaq::ExecutionResult());
    else
      throw std::runtime_error("Job completed but no result ID found.");
  }
  const std::string resultPath =
      resultType == QuantinuumServerHelper::ResultType::QSYS
          ? baseUrl + qsysResultsEndpoint + resultId
          : baseUrl + resultsEndpoint + resultId;
  CUDAQ_INFO("Retrieving results from path: {}", resultPath);
  RestHeaders headers = generateRequestHeader();
  RestCookies cookies = getCookies();
  // If this is a Qsys result, use the default version to retrieve accumulated
  // shot data.
  const std::string paramStr =
      resultType == QuantinuumServerHelper::ResultType::QSYS
          ? fmt::format("?version={}",
                        static_cast<int>(QsysResultVersion::DEFAULT))
          : std::string();

  // Retrieve the results
  auto resultResponse =
      restClient.get(resultPath, paramStr, headers, false, cookies);
  CUDAQ_INFO("Job result response: {}\n", resultResponse.dump());
  if (resultType == QuantinuumServerHelper::ResultType::PYTKET) {
    auto shotResults = resultResponse["data"]["attributes"]["shots"];
    CUDAQ_DBG("Count data: {}", shotResults.dump());

    // Get the register names
    auto bitResults = resultResponse["data"]["attributes"]["bits"];
    std::vector<std::string> outputNames;
    for (auto item : bitResults) {
      CUDAQ_DBG("Bit data: {}", item.dump());
      const auto registerName = item[0].get<std::string>();
      outputNames.push_back(registerName);
    }
    // The names are listed in the reverse order (w.r.t. CUDA-Q bit indexing
    // convention)
    std::reverse(outputNames.begin(), outputNames.end());
    return cudaq::utils::quantinuum::processResults(shotResults, outputNames);
  } else {
    const std::string qirResults =
        resultResponse["data"]["attributes"]["results"];
    CUDAQ_DBG("Count result data: {}", qirResults);

    return createSampleResultFromQirOutput(qirResults);
  }
}

// Extract QIR output data
std::string QuantinuumServerHelper::extractOutputLog(ServerMessage &jobResponse,
                                                     std::string &jobId) {
  const auto [resultType, resultId] = getResultId(jobResponse);
  if (resultId.empty()) {
    if (!jobReturnsResult(jobResponse)) {
      CUDAQ_INFO("Syntax checker job completed, no output to extract.");
      return "";
    } else {
      throw std::runtime_error("Job completed but no result ID found.");
    }
  }
  if (resultType != QuantinuumServerHelper::ResultType::QSYS) {
    throw std::runtime_error(
        "Expected QSYS result type for QIR output extraction.");
  }

  const std::string resultPath = baseUrl + qsysResultsEndpoint + resultId;
  CUDAQ_INFO("Retrieving results from path: {}", resultPath);
  RestHeaders headers = generateRequestHeader();
  RestCookies cookies = getCookies();
  // Retrieve the results (default version for QIR output)
  auto resultResponse = restClient.get(
      resultPath,
      fmt::format("?version={}", static_cast<int>(QsysResultVersion::DEFAULT)),
      headers, false, cookies);
  CUDAQ_INFO("Job result response: {}\n", resultResponse.dump());
  const std::string programType =
      resultResponse["data"]["relationships"]["program"]["data"]["type"]
          .get<std::string>();
  if (programType != "qir") {
    throw std::runtime_error(
        "Expected 'qir' type in the result response, got: " + programType);
  }

  const std::string qirResult =
      resultResponse["data"]["attributes"]["results"].get<std::string>();
  return qirResult;
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
    CUDAQ_INFO("Refreshing id-token");
    RestHeaders cookies{{"myqos_oat", refreshKey}};
    RestCookies headers = generateRequestHeader();
    nlohmann::json j;
    auto response_json = restClient.post(baseUrl, authEndpoint, j, headers,
                                         false, false, cookies, &cookies);
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

cudaq::ExtraPayloadProvider *QuantinuumServerHelper::getExtraPayloadProvider() {
  const auto extraPayloadProvider =
      runtimeTarget.runtimeConfig.find("extra_payload_provider");
  if (extraPayloadProvider == runtimeTarget.runtimeConfig.end())
    return nullptr;
  const auto &extraPayloadProviderName = extraPayloadProvider->second;

  auto &extraProviders = cudaq::getExtraPayloadProviders();
  const auto it = std::find_if(
      extraProviders.begin(), extraProviders.end(), [&](const auto &entry) {
        return entry->name() == extraPayloadProviderName;
      });
  if (it == extraProviders.end())
    throw std::runtime_error("ExtraPayloadProvider with name " +
                             extraPayloadProviderName + " not found.");
  CUDAQ_INFO("[QuantinuumServerHelper] Found extra payload provider '{}'.",
             extraPayloadProviderName);
  return it->get();
}

bool QuantinuumServerHelper::jobReturnsResult(
    ServerMessage &jobResponse) const {
  // Retrieve the device name if available.
  auto deviceNamePath =
      "/data/attributes/definition/backend_config/device_name"_json_pointer;
  if (!jobResponse.contains(deviceNamePath))
    return true;
  const std::string deviceName = jobResponse[deviceNamePath].get<std::string>();
  // Helios (NG device) syntax checker jobs won't return a result.
  if (deviceName.starts_with("Helios") && deviceName.ends_with("SC"))
    return false;

  return true;
}
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantinuumServerHelper,
                    quantinuum)
