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
#include <thread>
namespace cudaq {

/// @brief The OQCServerHelper class extends the ServerHelper class to handle
/// interactions with the OQC server for submitting and retrieving quantum
/// computation jobs.
class OQCServerHelper : public ServerHelper {
private:
  /// @brief RestClient used for HTTP requests.

  /// @brief Retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key) const;

  /// @brief Check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  /// @brief Create n requested tasks placeholders returning uuids for each
  std::vector<std::string> createNTasks(int n);

  /// @brief make a compiler config json string parameterising with number of
  /// shots
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
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) override;

  std::string get_from_config(BackendConfig config, const std::string &key,
                              const std::string &default_return = "");
};

// Initialize the OQC server helper with a given backend configuration
void OQCServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing OQC Backend.");
  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
  // Set the necessary configuration variables for the OQC API
  backendConfig["url"] = OQCServerHelper::get_from_config(
      config, "url", "https://sandbox.qcaas.oqc.app");
  backendConfig["version"] = "v0.3";
  backendConfig["user_agent"] = "cudaq/0.3.0";
  backendConfig["target"] = "Lucy";
  backendConfig["qubits"] = 8;
  backendConfig["email"] = OQCServerHelper::get_from_config(config, "email");
  backendConfig["password"] =
      OQCServerHelper::get_from_config(config, "password");
  // Retrieve the email/password key from the environment variables
  if (backendConfig["email"].empty())
    backendConfig["email"] = getEnvVar("OQC_EMAIL");
  if (backendConfig["password"].empty())
    backendConfig["password"] = getEnvVar("OQC_PASSWORD");

  // Construct the API job path
  backendConfig["job_path"] = "/tasks"; // backendConfig["url"] + "/tasks";
}

std::string
OQCServerHelper::get_from_config(BackendConfig config, const std::string &key,
                                 const std::string &default_return) {
  auto iter = backendConfig.find(key);
  return iter != backendConfig.end() ? iter->second : default_return;
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

std::vector<std::string> OQCServerHelper::createNTasks(int n) {
  RestHeaders headers = OQCServerHelper::getHeaders();
  nlohmann::json j;
  std::vector<std::string> output;
  for (int i = 0; i < n; ++i) {
    auto response = client.post(backendConfig.at("url"),
                                backendConfig.at("job_path"), j, headers);
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
      OQCServerHelper::createNTasks(static_cast<int>(circuitCodes.size()));

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
OQCServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobId) {

  if (postJobResponse["results"].is_null()) {
    RestHeaders headers = getHeaders();
    auto errorPath = backendConfig.at("url") + backendConfig["job_path"] + "/" +
                     jobId + "/error";
    cudaq::info(
        "Null results received; fetching detailed error message here: {}",
        errorPath);
    ServerMessage errorMessage = client.get(errorPath, "", headers);
    throw std::runtime_error("OQC backend error message: " +
                             errorMessage.dump());
  }
  cudaq::info("postJobResponse is {}", postJobResponse.dump());
  const auto &jsonResults = postJobResponse.at("results");

  cudaq::sample_result sampleResult; // value to return
  if (jsonResults.is_array() && jsonResults.size() == 0)
    throw std::runtime_error("No measurements found. Were any in the program?");

  // Try to determine between two results formats:
  //   {"results":{"r0_r1_r2_r3_r4":{"00000":1000}}} (hasResultNames = true )
  //   {"results":{"00":479,"11":521}}               (hasResultNames = false)
  bool hasResultNames = false;
  for (const auto &element : jsonResults.items()) {
    // element.value() is either something like
    // {"00000":1000}} or 479
    if (element.value().is_object()) {
      hasResultNames = true;
      break;
    }
  }

  CountsDictionary countsDict;
  if (hasResultNames) {
    // The following code only supports 1 object in the returned results because
    // there is only 1 CountsDictionary and 1 register name in use, so throw a
    // warning if that isn't true.
    if (jsonResults.size() != 1)
      cudaq::info("WARNING: unexpected jsonResults size ({}). Continuing to "
                  "parse anyway.",
                  jsonResults.size());

    // Note: `name` contains a concatenated list of measurement names as
    // specified in the sent QIR program, separated by underscores.  A
    // potential future enhancement would be to make separate
    // ExecutionResult's for each register.
    // Example jsonResults: {"r0_r1_r2_r3_r4":{"00000":1000,"11111":1000}}
    for (const auto &[name, counts] : jsonResults.items()) {
      for (auto &element : counts.items())
        countsDict[element.key()] = element.value();
      cudaq::ExecutionResult executionResult{counts, cudaq::GlobalRegisterName};
      sampleResult.append(executionResult);
    }
  } else {
    // Example jsonResults: {"00":479,"11":521}
    for (auto &element : jsonResults.items())
      countsDict[element.key()] = element.value();
    cudaq::ExecutionResult executionResult{countsDict,
                                           cudaq::GlobalRegisterName};
    sampleResult.append(executionResult);
  }
  return sampleResult;
}

// Get the headers for the API requests
RestHeaders OQCServerHelper::getHeaders() {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("email") || !keyExists("password"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  // Construct the headers
  RestHeaders headers;
  headers["Content-Type"] = "application/json";

  nlohmann::json j;
  j["email"] = backendConfig.at("email");
  j["password"] = backendConfig.at("password");
  nlohmann::json response =
      client.post(backendConfig.at("url") + "/auth", "", j, headers,
                  /*enableLossgging=*/false);
  std::string key = response.at("access_token");
  backendConfig["access_token"] = key;

  headers["Authorization"] = "Bearer " + backendConfig["access_token"];

  // Return the headers
  return headers;
}

} // namespace cudaq

// Register the OQC server helper in the CUDAQ server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::OQCServerHelper, oqc)
