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
#include <bitset>
#include <fstream>
#include <map>
#include <thread>

namespace cudaq {

/// @brief The IonQServerHelper class extends the ServerHelper class to handle
/// interactions with the IonQ server for submitting and retrieving quantum
/// computation jobs.
class IonQServerHelper : public ServerHelper {
private:
  /// @brief RestClient used for HTTP requests.
  RestClient client;

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key) const;

  /// @brief Helper method to check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  /// @brief Output names indexed by jobID/taskID
  std::map<std::string, OutputNamesType> outputNames;

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
};

// Initialize the IonQ server helper with a given backend configuration
void IonQServerHelper::initialize(BackendConfig config) {
  cudaq::info("Initializing IonQ Backend.");
  // Move the passed config into the member variable backendConfig
  // Set the necessary configuration variables for the IonQ API
  backendConfig["url"] = config.find("url") != config.end()
                             ? config["url"]
                             : "https://api.ionq.co";
  backendConfig["version"] = "v0.3";
  backendConfig["user_agent"] = "cudaq/0.3.0";
  backendConfig["target"] =
      config.find("qpu") != config.end() ? config["qpu"] : "simulator";
  backendConfig["qubits"] = 29;
  // Retrieve the noise model setting (if provided)
  if (config.find("noise") != config.end())
    backendConfig["noise_model"] = config["noise"];
  // Retrieve the API key from the environment variables
  backendConfig["token"] = getEnvVar("IONQ_API_KEY");
  // Construct the API job path
  backendConfig["job_path"] =
      backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  // Parse the output_names.* (for each job) and place it in outputNames[]
  for (auto &[key, val] : config) {
    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToBaseProfileQIR.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      for (const auto &el : outputNamesJSON[0]) {
        std::size_t result = el[0].get<std::size_t>();
        std::size_t qubit = el[1][0].get<std::size_t>();
        std::string registerName = el[1][1].get<std::string>();
        jobOutputNames[result] = {qubit, registerName};
      }

      this->outputNames[key] = jobOutputNames;
    }
  }
}

// Retrieve an environment variable
std::string IonQServerHelper::getEnvVar(const std::string &key) const {
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
    job["target"] = backendConfig.at("target");
    // Add noise model config to the JSON job request if a noise model was set
    // and the IonQ 'simulator' target was selected.
    if (keyExists("noise_model") && backendConfig.at("target") == "simulator") {
      nlohmann::json noiseModel;
      noiseModel["model"] = backendConfig.at("noise_model");
      job["noise"] = noiseModel;
    }

    job["qubits"] = backendConfig.at("qubits");
    job["shots"] = static_cast<int>(shots);
    job["input"]["format"] = "qir";
    job["input"]["data"] = circuitCode.code;
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

// Construct the path to get the results of a job
std::string
IonQServerHelper::constructGetResultsPath(ServerMessage &postResponse) {
  // Check if the necessary keys exist in the response and the configuration
  if (!postResponse.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = postResponse.at("jobs");

  if (jobs.empty() || !jobs[0].contains("results_url"))
    throw std::runtime_error(
        "ServerMessage doesn't contain 'results_url' key in the first job.");

  if (!keyExists("url"))
    throw std::runtime_error("Key 'url' doesn't exist in backendConfig.");

  // Return the results path
  return backendConfig.at("url") + jobs[0].at("results_url").get<std::string>();
}

// Overloaded version of constructGetResultsPath for jobId input
std::string IonQServerHelper::constructGetResultsPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  // Return the results path
  return backendConfig.at("job_path") + jobId + "/results";
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
    throw std::runtime_error(
        "The job failed upon submission. Check the job submission in your IonQ "
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

  // Get the number of qubits. This assumes the all qubits are measured, which
  // is a safe assumption for now but may change in the future.
  cudaq::debug("postJobResponse message: {}", postJobResponse.dump());
  auto &jobs = postJobResponse.at("jobs");
  if (!jobs[0].contains("qubits"))
    throw std::runtime_error(
        "ServerMessage doesn't tell us how many qubits there were");

  auto nQubits = jobs[0].at("qubits").get<int>();
  cudaq::debug("nQubits is : {}", nQubits);
  cudaq::debug("Results message: {}", results.dump());

  if (outputNames.find("output_names." + jobID) == outputNames.end()) {
    throw std::runtime_error("Could not find output names for job " + jobID +
                             " this " + std::to_string((long)this));
  }

  auto &output_names = outputNames["output_names." + jobID];
  for (auto &[result, info] : output_names) {
    cudaq::info("Qubit {} Result {} Name {}", info.qubitNum, result,
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

  // Make a map sorted by register name ([str] --> <qubit,result>)
  std::map<std::string, std::pair<std::size_t, std::size_t>> mapResultStr;
  for (const auto &[result, info] : output_names)
    mapResultStr[info.registerName] = std::make_pair(info.qubitNum, result);

  // Get a reduced list of qubit numbers that were in the original program so
  // that we can slice the output data and extract the bits that the user was
  // interested in.
  std::vector<std::size_t> qubitNumbers;
  qubitNumbers.reserve(output_names.size());
  for (const auto &[regName, qubitAndResult] : mapResultStr)
    qubitNumbers.push_back(qubitAndResult.first);

  // For each original counts entry in the full sample results, reduce it down
  // to the user component and add to userGlobal. This is similar to
  // `get_marginal()`, but that sorts the input indices, which isn't necessarily
  // desirable here.
  CountsDictionary userGlobal;
  for (const auto &[bits, count] : fullSampleResults) {
    std::string userBitStr;
    for (const auto &qubit : qubitNumbers) {
      if (qubit < bits.size())
        userBitStr += bits[qubit];
      else
        throw std::runtime_error(fmt::format(
            "Cannot fetch qubit index {} from bits '{}'; bits.size() = {}",
            qubit, bits, bits.size()));
    }
    userGlobal[userBitStr] += count;
  }
  execResults.emplace_back(userGlobal);

  // Now add to `execResults` one register at a time
  for (const auto &[regName, qubitAndResult] : mapResultStr) {
    CountsDictionary regCounts;
    for (const auto &[bits, count] : fullSampleResults)
      regCounts[std::string{bits[qubitAndResult.first]}] += count;
    execResults.emplace_back(regCounts, regName);
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

// Register the IonQ server helper in the CUDAQ server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
