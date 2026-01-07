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
#include <map>
#include <thread>

namespace cudaq {

/// @brief The IonQServerHelper class extends the ServerHelper class to handle
/// interactions with the IonQ server for submitting and retrieving quantum
/// computation jobs.
class IonQServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api.ionq.co";
  static constexpr const char *DEFAULT_VERSION = "v0.3";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "ionq"; }

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
                                      std::string &jobId) override;

private:
  /// @brief RestClient used for HTTP requests.
  RestClient client;

  /// @brief Helper method to set the number of qubits based on the target.
  int setQubits(const std::string &target);

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

// Initialize the IonQ server helper with a given backend configuration
void IonQServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing IonQ Backend.");
  // Move the passed config into the member variable backendConfig
  // Set the necessary configuration variables for the IonQ API
  backendConfig["url"] = getValueOrDefault(config, "url", DEFAULT_URL);
  backendConfig["version"] = DEFAULT_VERSION;
  backendConfig["user_agent"] = "cudaq/" + std::string(cudaq::getVersion());
  backendConfig["target"] = getValueOrDefault(config, "qpu", "simulator");
  backendConfig["qubits"] = setQubits(backendConfig["target"]);
  // Retrieve the noise model setting (if provided)
  if (config.find("noise") != config.end())
    backendConfig["noise_model"] = config["noise"];
  // Retrieve the API key from the environment variables
  bool isTokenRequired = [&]() {
    auto it = config.find("emulate");
    if (it != config.end() && it->second == "true")
      return false;
    return true;
  }();
  backendConfig["token"] = getEnvVar("IONQ_API_KEY", "0", isTokenRequired);
  // Construct the API job path
  backendConfig["job_path"] =
      backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  parseConfigForCommonParams(config);

  // Enable debiasing
  if (config.find("debias") != config.end())
    backendConfig["debias"] = config["debias"];
  if (config.find("sharpen") != config.end())
    backendConfig["sharpen"] = config["sharpen"];
  if (config.find("format") != config.end())
    backendConfig["format"] = config["format"];
}

// Implementation of the getValueOrDefault function
std::string
IonQServerHelper::getValueOrDefault(const BackendConfig &config,
                                    const std::string &key,
                                    const std::string &defaultValue) const {
  return config.find(key) != config.end() ? config.at(key) : defaultValue;
}

// Retrieve an environment variable
std::string IonQServerHelper::getEnvVar(const std::string &key,
                                        const std::string &defaultVal,
                                        const bool isRequired) const {
  // Get the environment variable
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, either return the default or throw an exception
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error(key + " environment variable is not set.");
    else
      return defaultVal;
  }
  // Return the variable as a string
  return std::string(env_var);
}

// Helper function to get a value from a dictionary or return a default
template <typename K, typename V>
V getOrDefault(const std::unordered_map<K, V> &map, const K &key,
               const V &defaultValue) {
  auto it = map.find(key);
  return (it != map.end()) ? it->second : defaultValue;
}

// Set the number of qubits based on the target
int IonQServerHelper::setQubits(const std::string &target) {
  static const std::unordered_map<std::string, int> qubitMap = {
      {"simulator", 29},
      {"qpu.harmony", 11},
      {"qpu.aria-1", 25},
      {"qpu.forte-enterprise-1", 36},
      {"qpu.forte-enterprise-2", 36}};

  return getOrDefault(qubitMap, target, 29); // 29 is the default value
}

// Check if a key exists in the backend configuration
bool IonQServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

// Create a job for the IonQ quantum computer
ServerJobPayload
IonQServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("target") || !keyExists("qubits") || !keyExists("job_path"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  std::vector<ServerMessage> jobs;
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage job;
    job["name"] = circuitCode.name;
    job["target"] = backendConfig.at("target");
    // Add noise model config to the JSON job request if a noise model was
    // set and the IonQ 'simulator' target was selected.
    if (keyExists("noise_model") && backendConfig.at("target") == "simulator") {
      nlohmann::json noiseModel;
      noiseModel["model"] = backendConfig.at("noise_model");
      job["noise"] = noiseModel;
    }

    job["qubits"] = backendConfig.at("qubits");
    job["shots"] = shots;
    job["input"]["format"] = "qir";
    job["input"]["data"] = circuitCode.code;
    // Include error mitigation configuration if set in backendConfig
    if (keyExists("debias")) {
      try {
        bool debiasValue =
            nlohmann::json::parse(backendConfig["debias"]).get<bool>();
        job["error_mitigation"]["debias"] = debiasValue;
      } catch (const nlohmann::json::exception &e) {
        throw std::runtime_error(
            "Invalid value for 'debias'. It should be a boolean (true/false).");
      }
    }

    jobs.push_back(job);
  }

  // Return a tuple containing the job path, headers, and the job message
  auto ret = std::make_tuple(backendConfig.at("job_path"), getHeaders(), jobs);
  return ret;
}

// From a server message, extract the job ID
std::string IonQServerHelper::extractJobId(ServerMessage &postResponse) {
  // If the response does not contain the key 'id', throw an exception
  if (!postResponse.contains("id"))
    throw std::runtime_error("ServerMessage doesn't contain 'id' key.");

  // Return the job ID from the response
  auto ret = postResponse.at("id");
  return ret;
}

// Construct the path to get a job
std::string IonQServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  // Check if the necessary keys exist in the response and the configuration
  if (!postResponse.contains("results_url"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'results_url' key.");

  if (!keyExists("url"))
    throw std::runtime_error("Key 'url' doesn't exist in backendConfig.");

  // Return the job path
  auto ret = backendConfig.at("url") +
             postResponse.at("results_url").get<std::string>();
  return ret;
}

// Overloaded version of constructGetJobPath for jobId input
std::string IonQServerHelper::constructGetJobPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  // Return the job path
  auto ret = backendConfig.at("job_path") + "?id=" + jobId;
  return ret;
}

// Function to check if a URL already has query parameters
bool hasQueryParameters(const std::string &url) {
  return url.find("?") != std::string::npos;
}

// Function to append a query parameter to a URL
void appendQueryParam(std::string &url, const std::string &param,
                      const std::string &value) {
  url += hasQueryParameters(url) ? "&" : "?";
  url += param + "=" + value;
}

// Construct the path to get the results of a job
std::string
IonQServerHelper::constructGetResultsPath(ServerMessage &postResponse) {
  // Check if the necessary keys exist in the response and the configuration
  if (!postResponse.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = postResponse.at("jobs");

  if (jobs.empty() || !jobs[0].contains("results_url"))
    throw std::runtime_error("ServerMessage doesn't contain 'results_url' "
                             "key in the first job.");

  if (!keyExists("url"))
    throw std::runtime_error("Key 'url' doesn't exist in backendConfig.");

  std::string resultsPath =
      backendConfig.at("url") + jobs[0].at("results_url").get<std::string>();

  // If sharpen is true, add it to the query parameters
  if (keyExists("sharpen") && backendConfig["sharpen"] == "true") {
    appendQueryParam(resultsPath, "sharpen", "true");
  }

  // Get specific results format
  if (keyExists("format")) {
    appendQueryParam(resultsPath, "format", backendConfig["format"]);
  } else {
    appendQueryParam(resultsPath, "format", "qir.quantum-log.v0");
  }

  return resultsPath;
}

// Overloaded version of constructGetResultsPath for jobId input
std::string IonQServerHelper::constructGetResultsPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  // Construct the results path
  std::string resultsPath = backendConfig.at("job_path") + jobId + "/results";

  // If sharpen is true, add it to the query parameters
  if (keyExists("sharpen") && backendConfig["sharpen"] == "true")
    resultsPath += "?sharpen=true";

  // Return the results path
  return resultsPath;
}

// Get the results from a given path
ServerMessage IonQServerHelper::getResults(std::string &resultsGetPath) {
  RestHeaders headers = getHeaders();
  // Return the results from the client
  return client.get(resultsGetPath, "", headers);
}

// Check if a job is done
bool IonQServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Check if the necessary keys exist in the response
  if (!getJobResponse.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = getJobResponse.at("jobs");

  if (jobs.empty() || !jobs[0].contains("status"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'status' key in the first job.");

  // Throw a runtime error if the job has failed
  if (jobs[0].at("status").get<std::string>() == "failed")
    throw std::runtime_error("The job failed upon submission. Check the "
                             "job submission in your IonQ "
                             "account for more information.");

  // Return whether the job is completed
  return jobs[0].at("status").get<std::string>() == "completed";
}

// Process the results from a job
cudaq::sample_result
IonQServerHelper::processResults(ServerMessage &postJobResponse,
                                 std::string &jobID) {
  // Construct the path to get the results
  auto resultsGetPath = constructGetResultsPath(postJobResponse);
  // Get the results
  auto results = getResults(resultsGetPath);

  // Get the number of qubits. This assumes the all qubits are measured,
  // which is a safe assumption for now but may change in the future.
  CUDAQ_DBG("postJobResponse message: {}", postJobResponse.dump());
  auto &jobs = postJobResponse.at("jobs");
  if (!jobs[0].contains("qubits"))
    throw std::runtime_error(
        "ServerMessage doesn't tell us how many qubits there were");

  auto nQubits = jobs[0].at("qubits").get<int>();
  CUDAQ_DBG("nQubits is : {}", nQubits);
  CUDAQ_DBG("Results message: {}", results.dump());

  if (outputNames.find(jobID) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobID);

  auto &output_names = outputNames[jobID];
  for (auto &[result, info] : output_names) {
    CUDAQ_INFO("Qubit {} Result {} Name {}", info.qubitNum, result,
               info.registerName);
  }

  cudaq::CountsDictionary counts;

  // Process the results
  assert(nQubits <= 64);
  for (const auto &element : results.items()) {
    // Convert base-10 ASCII key to bitstring and perform endian swap
    uint64_t s = std::stoull(element.key());
    std::string newkey = std::bitset<64>(s).to_string();
    std::reverse(newkey.begin(), newkey.end()); // perform endian swap
    newkey.resize(nQubits);

    double value = element.value().get<double>();
    std::size_t count = static_cast<std::size_t>(value * shots);
    counts[newkey] = count;
  }

  // Full execution results include compiler-generated qubits, which are
  // undesirable to the user.
  cudaq::ExecutionResult fullExecResults{counts};
  auto fullSampleResults = cudaq::sample_result{fullExecResults};

  // clang-format off
  // The following code strips out and reorders the outputs based on output_names.
  // For example, if `counts` is something like:
  //      { 11111:62 01111:12 11110:12 01110:12 }
  // And if we want to discard the first bit (because qubit 0 was a
  // compiler-generated qubit), that maps to something like this:
  // -----------------------------------------------------
  // Qubit  Index - x1234    x1234    x1234    x1234
  // Result Index - x0123    x0123    x0123    x0123
  //              { 11111:62 01111:12 11110:12 01110:12 }
  //              { x1111:62 x1111:12 x1110:12 x1110:12 }
  //                  \--- v ---/       \--- v ---/
  //              {    1111:(62+12)     x1110:(12+12)   }
  //              {    1111:74           1110:24        }
  // -----------------------------------------------------
  // clang-format on

  std::vector<ExecutionResult> execResults;

  // Get a reduced list of qubit numbers that were in the original program
  // so that we can slice the output data and extract the bits that the user
  // was interested in. Sort by QIR qubit number.
  std::vector<std::size_t> qubitNumbers;
  qubitNumbers.reserve(output_names.size());
  for (auto &[result, info] : output_names) {
    qubitNumbers.push_back(info.qubitNum);
  }

  // For each original counts entry in the full sample results, reduce it
  // down to the user component and add to userGlobal. If qubitNumbers is empty,
  // that means all qubits were measured.
  if (qubitNumbers.empty()) {
    execResults.emplace_back(ExecutionResult{fullSampleResults.to_map()});
  } else {
    auto subset = fullSampleResults.get_marginal(qubitNumbers);
    execResults.emplace_back(ExecutionResult{subset.to_map()});
  }

  // Now add to `execResults` one register at a time
  for (const auto &[result, info] : output_names) {
    CountsDictionary regCounts;
    for (const auto &[bits, count] : fullSampleResults)
      regCounts[std::string{bits[info.qubitNum]}] += count;
    execResults.emplace_back(regCounts, info.registerName);
  }

  // Return a sample result including the global register and all individual
  // registers.
  auto ret = cudaq::sample_result(execResults);
  return ret;
}

// Get the headers for the API requests
RestHeaders IonQServerHelper::getHeaders() {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("token") || !keyExists("user_agent"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  // Construct the headers
  RestHeaders headers;
  headers["Authorization"] = "apiKey " + backendConfig.at("token");
  headers["Content-Type"] = "application/json";
  headers["User-Agent"] = backendConfig.at("user_agent");

  // Return the headers
  return headers;
}

} // namespace cudaq

// Register the IonQ server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
