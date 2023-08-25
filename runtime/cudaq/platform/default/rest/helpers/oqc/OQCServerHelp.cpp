/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <thread>
namespace cudaq {

/// @brief The OQCServerHelper class extends the ServerHelper class to handle
/// interactions with the OQC server for submitting and retrieving quantum
/// computation jobs.
class OQCServerHelper : public ServerHelper {
private:
  /// @brief RestClient used for HTTP requests.

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key) const;

  /// @brief Helper method to check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  std::vector<std::string> getJobID(int n);

  std::string makeConfig(int shots);

public:
  RestClient client;

  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "oqc"; }

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

  /// @brief Constructs the URL for retrieving the results of a job based on the
  /// server's response to a job submission.
  std::string constructGetResultsPath(ServerMessage &postResponse);

  /// @brief Constructs the URL for retrieving the results of a job based on a
  /// job ID.
  std::string constructGetResultsPath(std::string &jobId);

  /// @brief Retrieves the results of a job using the provided path.
  ServerMessage getResults(std::string &resultsGetPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &postJobResponse) override;

  std::string get_from_config(BackendConfig config, std::string key);
  std::string get_from_config(BackendConfig config, std::string key,
                              std::string default_return);
};

// Initialize the OQC server helper with a given backend configuration
void OQCServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing OQC Backend.");
  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
  // Set the necessary configuration variables for the OQC API
  backendConfig["url"] = OQCServerHelper::get_from_config(config, "url");
  backendConfig["version"] = "v0.3";
  backendConfig["user_agent"] = "cudaq/0.3.0";
  backendConfig["target"] = "Lucy";
  backendConfig["qubits"] = 8;
  // Retrieve the API key from the environment variables
  backendConfig["email"] = OQCServerHelper::get_from_config(config, "email");
  backendConfig["password"] =
      OQCServerHelper::get_from_config(config, "password");
  // Construct the API job path
  backendConfig["job_path"] = "/tasks"; // backendConfig["url"] + "/tasks";
}

std::string OQCServerHelper::get_from_config(BackendConfig config,
                                             std::string key,
                                             std::string default_return) {
  std::string output = OQCServerHelper::get_from_config(config, key);
  if (output.empty()) {
    return default_return;
  }
  return output;
}

std::string OQCServerHelper::get_from_config(BackendConfig config,
                                             std::string key) {
  auto iter = backendConfig.find(key);
  if (iter != backendConfig.end()) {
    return iter->second;
  }
  return "";
}

// Retrieve an environment variable
std::string OQCServerHelper::getEnvVar(const std::string &key) const {
  // Get the environment variable
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, throw an exception
  if (env_var == nullptr) {
    throw std::runtime_error(key + " environment variable is not set.");
  }
  // Return the variable as a string
  return std::string(env_var);
}

// Check if a key exists in the backend configuration
bool OQCServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

std::vector<std::string> OQCServerHelper::getJobID(int n) {
  RestHeaders headers = OQCServerHelper::getHeaders();
  nlohmann::json j;
  std::vector<std::string> output;
  for (int i = 0; i < n; ++i) {
    nlohmann::json_v3_11_1::json response = client.post(
        backendConfig.at("url"), backendConfig.at("job_path"), j, headers);
    output.push_back(response[0]);
  }
  return output;
}

std::string OQCServerHelper::makeConfig(int shots) {
  return "{\"$type\": \"<class 'scc.compiler.config.CompilerConfig'>\", "
         "\"$data\": {\"repeats\": " +
         std::to_string(shots) +
         ", \"repetition_period\": null, \"results_format\": {\"$type\": "
         "\"<class 'scc.compiler.config.QuantumResultsFormat'>\", \"$data\": "
         "{\"format\": {\"$type\": \"<enum "
         "'scc.compiler.config.InlineResultsProcessing'>\", \"$value\": 1}, "
         "\"transforms\": {\"$type\": \"<enum "
         "'scc.compiler.config.ResultsFormatting'>\", \"$value\": 3}}}, "
         "\"metrics\": {\"$type\": \"<enum "
         "'scc.compiler.config.MetricsType'>\", \"$value\": 6}, "
         "\"active_calibrations\": [], \"optimizations\": {\"$type\": \"<class "
         "'scc.compiler.config.Tket'>\", \"$data\": {\"tket_optimizations\": "
         "{\"$type\": \"<enum 'scc.compiler.config.TketOptimizations'>\", "
         "\"$value\": 30}}}}}";
}

// Create a job for the OQC quantum computer
ServerJobPayload
OQCServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("target") || !keyExists("qubits") || !keyExists("job_path"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");
  std::vector<ServerMessage> jobs(circuitCodes.size());
  std::vector<std::string> task_ids =
      OQCServerHelper::getJobID(static_cast<int>(circuitCodes.size()));

  for (size_t i = 0; i < circuitCodes.size(); ++i) {
    nlohmann::json j;
    j["tasks"] = std::vector<nlohmann::json>();
    // Construct the job message
    nlohmann::json job;
    job["task_id"] = task_ids[i];
    job["config"] = makeConfig(static_cast<int>(shots));
    job["program"] = circuitCodes[i].code;
    j["tasks"].push_back(job);
    jobs[i] = j;
  }

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(backendConfig.at("url") +
                             backendConfig.at("job_path") + "/submit",
                         getHeaders(), jobs);
}

// From a server message, extract the job ID
std::string OQCServerHelper::extractJobId(ServerMessage &postResponse) {
  // If the response does not contain the key 'id', throw an exception
  if (!postResponse.contains("task_id"))
    return "";

  // Return the job ID from the response
  return postResponse.at("task_id");
}

// Construct the path to get a job
std::string OQCServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return backendConfig.at("job_path") + "/" +
         postResponse.at("task_id").get<std::string>() + "/results";
}

// Overloaded version of constructGetJobPath for jobId input
std::string OQCServerHelper::constructGetJobPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  // Return the job path
  return backendConfig.at("url") + backendConfig.at("job_path") + "/" + jobId +
         "/results";
}

// Construct the path to get the results of a job
std::string
OQCServerHelper::constructGetResultsPath(ServerMessage &postResponse) {
  // Return the results path
  return backendConfig.at("job_path") + "/" +
         postResponse.at("task_id").get<std::string>() + "/results";
}

// Overloaded version of constructGetResultsPath for jobId input
std::string OQCServerHelper::constructGetResultsPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  // Return the results path
  return backendConfig.at("job_path") + "/" + jobId + "/results";
}

// Get the results from a given path
ServerMessage OQCServerHelper::getResults(std::string &resultsGetPath) {
  RestHeaders headers = getHeaders();
  // Return the results from the client
  return client.get(resultsGetPath, "", headers);
}

// Check if a job is done
bool OQCServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Check if the necessary keys exist in the response
  if (!getJobResponse.contains("results"))
    throw std::runtime_error("ServerMessage doesn't contain 'results' key.");

  // Return whether the job is completed
  return getJobResponse.at("results") != NULL;
}

// Process the results from a job
cudaq::sample_result
OQCServerHelper::processResults(ServerMessage &postJobResponse) {

  cudaq::CountsDictionary counts = postJobResponse.at("results");
  // Create an execution result
  cudaq::ExecutionResult executionResult(counts);
  // Return a sample result
  return cudaq::sample_result(executionResult);
}

// Get the headers for the API requests
RestHeaders OQCServerHelper::getHeaders() {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("email") || !keyExists("password"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  // Construct the headers
  RestHeaders headers;

  nlohmann::json j;
  j["email"] = backendConfig.at("email");

  j["password"] = backendConfig.at("password");

  nlohmann::json response =
      client.post(backendConfig.at("url") + "/auth", "", j, headers);

  std::string key = response.at("access_token");

  headers["Authorization"] = "Bearer " + key;
  headers["Content-Type"] = "application/json";
  // Return the headers
  return headers;
}

} // namespace cudaq

// Register the OQC server helper in the CUDAQ server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::OQCServerHelper, oqc)
