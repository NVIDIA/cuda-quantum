/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include <bitset>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_set>
using json = nlohmann::json;

bool isValidTarget(const std::string &input) {
  static const std::unordered_set<std::string> validTargets = {
      "cq_sqale_qpu", "cq_sqale_simulator"};
  return validTargets.find(input) != validTargets.end();
}

std::string formatOpenQasm(const std::string &input_string) {
  std::string escaped_string =
      std::regex_replace(input_string, std::regex("\\\\"), "\\\\");
  escaped_string = std::regex_replace(escaped_string, std::regex("\""), "\\\"");
  escaped_string = std::regex_replace(escaped_string, std::regex("\n"), "\\n");

  std::ostringstream json_stream;
  json_stream << "[\"" << escaped_string << "\"]";
  return json_stream.str();
}

namespace cudaq {

/// @brief The InfleqtionServerHelper class extends the ServerHelper class to
/// handle interactions with the Infleqtion server for submitting and retrieving
/// quantum computation jobs.
class InfleqtionServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://superstaq.infleqtion.com";
  static constexpr const char *DEFAULT_VERSION = "v0.2.0";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "infleqtion"; }

  /// @brief Returns the headers for the server requests.
  RestHeaders getHeaders() override;

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override;

  /// @brief Override the polling interval method
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(1);
  }

private:
  /// @brief RestClient used for HTTP requests.
  RestClient client;

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key, const std::string &defaultVal,
                        const bool isRequired) const;

  /// @brief Helper function to get value from config or return a default value.
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const;

  /// @brief Helper method to check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;
};

// Initialize the Infleqtion server helper with a given backend configuration
void InfleqtionServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing Infleqtion Backend.");

  // Move the passed config into the member variable backendConfig
  backendConfig = config;

  // Set default URL and version if not provided
  backendConfig["url"] = getValueOrDefault(config, "url", DEFAULT_URL);
  backendConfig["version"] =
      getValueOrDefault(config, "version", DEFAULT_VERSION);
  backendConfig["user_agent"] = getValueOrDefault(
      config, "user_agent", "cudaq/" + std::string(cudaq::getVersion()));
  backendConfig["machine"] =
      getValueOrDefault(config, "machine", "cq_sqale_simulator");
  backendConfig["method"] = config["method"];

  // Validate machine name client-side
  if (!isValidTarget(backendConfig["machine"])) {
    throw std::runtime_error("Invalid Infleqtion machine specified.");
  }

  // Determine if token is required
  bool isTokenRequired = [&]() {
    auto it = config.find("emulate");
    return !(it != config.end() && it->second == "true");
  }();

  // Get the API token from environment variable if not provided
  if (!keyExists("token")) {
    backendConfig["token"] =
        getEnvVar("SUPERSTAQ_API_KEY", "0", isTokenRequired);
  }

  // Construct the API job path
  backendConfig["job_path"] =
      backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";

  // Set shots if provided
  if (config.find("shots") != config.end())
    this->setShots(std::stoul(config["shots"]));

  // Parse common parameters
  parseConfigForCommonParams(config);
}

// Get the headers for the API requests
RestHeaders InfleqtionServerHelper::getHeaders() {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("token") || !keyExists("user_agent"))
    throw std::runtime_error(
        "Required keys 'token' or 'user_agent' not found in backendConfig.");

  // Construct the headers
  RestHeaders headers;
  headers["Authorization"] = backendConfig.at("token");
  headers["Content-Type"] = "application/json";
  headers["User-Agent"] = backendConfig.at("user_agent");

  // Return the headers
  return headers;
}

// Create a job for the Infleqtion quantum computer
ServerJobPayload
InfleqtionServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("machine") || !keyExists("job_path"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  auto &circuitCode = circuitCodes[0];

  // Construct the job message
  ServerMessage job;
  job["qasm_strs"] = formatOpenQasm(
      circuitCode.code); // Assuming code is in OpenQASM 2.0 format
  job["target"] = backendConfig.at("machine");
  job["shots"] = shots;

  if (backendConfig.count("method"))
    job["method"] = backendConfig.at("method");

  // Store output names and reorder indices if necessary
  OutputNamesType outputNamesMap;
  for (auto &item : circuitCode.output_names.items()) {
    std::size_t idx = std::stoul(item.key());
    ResultInfoType info;
    info.qubitNum = idx;
    info.registerName = item.value();
    outputNamesMap[idx] = info;
  }
  outputNames[circuitCode.name] = outputNamesMap;
  reorderIdx[circuitCode.name] = circuitCode.mapping_reorder_idx;

  // Prepare headers
  RestHeaders headers = getHeaders();

  // Return a tuple containing the job path, headers, and the job message
  auto ret = std::make_tuple(backendConfig.at("job_path"), headers,
                             std::vector<ServerMessage>{job});
  return ret;
}

// Extract the job ID from the server's response
std::string InfleqtionServerHelper::extractJobId(ServerMessage &postResponse) {
  // Check if the response contains 'job_ids' key
  if (!postResponse.contains("job_ids") || !postResponse["job_ids"].is_array())
    throw std::runtime_error("ServerMessage doesn't contain 'job_ids' key.");

  // Extract job ID from list response
  std::string jobId = postResponse["job_ids"][0];
  // Return the job ID
  return jobId;
}

// Construct the path to get a job based on the server's response
std::string
InfleqtionServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  // Extract job ID
  std::string jobId = extractJobId(postResponse);

  // Construct the path
  return constructGetJobPath(jobId);
}

// Construct the path to get a job based on job ID
std::string InfleqtionServerHelper::constructGetJobPath(std::string &jobId) {
  if (!keyExists("url") || !keyExists("version"))
    throw std::runtime_error(
        "Keys 'url' or 'version' don't exist in backendConfig.");

  // Construct the job path
  std::string jobPath = backendConfig.at("url") + '/' +
                        backendConfig.at("version") + "/job/" + jobId;
  return jobPath;
}

// Check if a job is done
bool InfleqtionServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Check if the response contains 'status' key
  if (!getJobResponse.contains("status"))
    throw std::runtime_error("ServerMessage is missing job 'status' key.");

  std::string status = getJobResponse["status"];
  if (status == "Canceled") {
    throw std::runtime_error("The submitted job was canceled.");
  }
  // Returns whether the job is done
  return status == "Done";
}

// Process the results from a job
cudaq::sample_result
InfleqtionServerHelper::processResults(ServerMessage &getJobResponse,
                                       std::string &jobId) {
  // Check if the response contains 'samples' key
  if (!getJobResponse.contains("samples"))
    throw std::runtime_error("Samples not found in the job results.");

  // Extract samples
  auto samplesJson = getJobResponse["samples"];
  cudaq::CountsDictionary counts;
  for (auto &item : samplesJson.items()) {
    std::string bitstring = item.key();
    std::size_t count = item.value();
    counts[bitstring] = count;
  }
  // Create an ExecutionResult
  cudaq::ExecutionResult execResult{counts};

  // Return the sample_result
  return cudaq::sample_result{execResult};
}

// Helper method to retrieve an environment variable
std::string InfleqtionServerHelper::getEnvVar(const std::string &key,
                                              const std::string &defaultVal,
                                              const bool isRequired) const {
  const char *env_var = std::getenv(key.c_str());
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error("Environment variable " + key +
                               " is required but not set.");
    else
      return defaultVal;
  }
  return std::string(env_var);
}

// Helper function to get a value from config or return a default
std::string InfleqtionServerHelper::getValueOrDefault(
    const BackendConfig &config, const std::string &key,
    const std::string &defaultValue) const {
  auto it = config.find(key);
  return (it != config.end()) ? it->second : defaultValue;
}

// Check if a key exists in the backend configuration
bool InfleqtionServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

} // namespace cudaq

// Register the Infleqtion server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::InfleqtionServerHelper,
                    infleqtion)
