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

namespace cudaq {

/// @brief The QuantumMachinesServerHelper class extends the ServerHelper class
/// to handle interactions with the Quantum Machines server for submitting and
/// retrieving quantum computation jobs.
class QuantumMachinesServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api.quantum-machines.com";
  static constexpr const char *DEFAULT_VERSION = "v1.0.0";
  static constexpr const char *DEFAULT_EXECUTOR = "mock";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "quantum_machines"; }

  /// @brief Returns the headers for the server requests.
  RestHeaders getHeaders() override {
    RestHeaders headers;
    headers["Content-Type"] = "application/json";
    headers["X-API-Key"] = backendConfig["api_key"];
    return headers;
  }

  // Helper function to get a value from config or return a default
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const {
    auto it = config.find(key);
    return (it != config.end()) ? it->second : defaultValue;
  }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override {
    CUDAQ_INFO("Initializing Quantum Machines Backend");
    backendConfig = config;
    backendConfig["url"] = getValueOrDefault(config, "url", DEFAULT_URL);
    backendConfig["version"] =
        getValueOrDefault(config, "version", DEFAULT_VERSION);
    backendConfig["executor"] =
        getValueOrDefault(config, "executor", DEFAULT_EXECUTOR);
    // Check for API key in config, then fall back to environment variable
    std::string apiKey = getValueOrDefault(config, "api_key", "");
    if (apiKey.empty()) {
      char *envApiKey = std::getenv("QUANTUM_MACHINES_API_KEY");
      if (envApiKey)
        apiKey = envApiKey;
    }
    backendConfig["api_key"] = apiKey;

    CUDAQ_INFO("Initializing Quantum Machines Backend. config: {}",
               backendConfig);
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  // A Server Job Payload consists of a job post URL path, the headers,
  // and a vector of related Job JSON messages.
  // using ServerJobPayload =
  //    std::tuple<std::string, RestHeaders, std::vector<ServerMessage>>;
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    CUDAQ_INFO("In createJob. code: {}", circuitCodes[0].code);
    ServerMessage job;
    job["content"] = circuitCodes[0].code;
    job["source"] = "oq2";
    job["shots"] = shots;
    job["executor"] = backendConfig["executor"];
    RestHeaders headers = getHeaders();
    std::string path = backendConfig["url"] + "/v1/execute";
    return std::make_tuple(path, headers, std::vector<ServerMessage>{job});
  }

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override {
    CUDAQ_INFO("In extractJobId. {}", postResponse.dump());
    if (!postResponse.contains("id"))
      return "";

    // Return the job ID from the response
    return postResponse.at("id");
  }

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // CUDAQ_INFO("In constructGetJobPath(postResponse)");
    CUDAQ_INFO("In constructGetJobPath(postResponse={})", postResponse.dump());
    return extractJobId(postResponse);
  }

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    CUDAQ_INFO("In constructGetJobPath(std::string &jobId)");
    std::string results_url = backendConfig["url"] + "/v1/results/" + jobId;
    return results_url;
  }

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override {
    CUDAQ_INFO("jobIsDone");
    std::string status = getJobResponse.at("status");
    return status == "Done" || status == "Failed";
  }

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override {
    CUDAQ_INFO("Sample results: {}", getJobResponse.dump());
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

  /// @brief Override the polling interval method
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    CUDAQ_INFO("nextResultPollingInterval");
    return std::chrono::seconds(1);
  }
};

} // namespace cudaq

// Register the Quantum Machines server helper in the CUDA-Q server helper
// factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantumMachinesServerHelper,
                    quantum_machines)
