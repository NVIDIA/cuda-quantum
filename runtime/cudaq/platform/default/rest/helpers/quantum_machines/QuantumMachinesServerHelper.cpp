/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nlohmann/json.hpp"
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
class QuantumMachinesServerHelper : public ServerHelper, public QirServerHelper  {
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
    backendConfig["qubit_mapping_mode"] =
        getValueOrDefault(config, "qubit_mapping_mode", "local_get_latest");
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

  inline std::string kernelExecutionToString(const KernelExecution &ke) {
    std::ostringstream ss;
    ss << "KernelExecution {\n";
    ss << "  name: " << ke.name << "\n";
    ss << "  code:\n" << ke.code << "\n";

    ss << "  jit: " << (ke.jit.has_value() ? "<present>" : "<none>") << "\n";
    ss << "  resourceCounts: "
      << (ke.resourceCounts.has_value() ? "<present>" : "<none>") << "\n";

    ss << "  output_names: " << ke.output_names.get().dump(2) << "\n";
    ss << "  user_data: " << ke.user_data.get().dump(2) << "\n";

    ss << "  mapping_reorder_idx: [";
    for (std::size_t i = 0; i < ke.mapping_reorder_idx.size(); ++i)
      ss << ke.mapping_reorder_idx[i]
        << (i + 1 < ke.mapping_reorder_idx.size() ? ", " : "");
    ss << "]\n";

    ss << "}";
    return ss.str();
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  // A Server Job Payload consists of a job post URL path, the headers,
  // and a vector of related Job JSON messages.
  // using ServerJobPayload =
  //    std::tuple<std::string, RestHeaders, std::vector<ServerMessage>>;
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    CUDAQ_INFO("In createJob. code: {}", kernelExecutionToString(circuitCodes[0]));
    auto *executionContext = cudaq::getExecutionContext();
    const std::string requestType =
        executionContext ? executionContext->name : "unknown";
    const bool isRunRequest = requestType == "run";
    ServerMessage job;
    job["content"] = circuitCodes[0].code;
    job["source"] = "quake";
    job["shots"] = shots;
    job["executor"] = backendConfig["executor"];
    job["qubit_mapping_mode"] = backendConfig["qubit_mapping_mode"];
    job["output_format"] = isRunRequest ? "qir-raw" : "histogram";
    RestHeaders headers = getHeaders();
    std::string path = backendConfig["url"] + "/v1/execute";
    return std::make_tuple(path, headers, std::vector<ServerMessage>{job});
  }

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override {
    CUDAQ_INFO("In extractJobId. {}", postResponse.dump());
    if (!postResponse.contains("id"))
      return "";

    // Extract the job ID from the response
    std::string id = postResponse.at("id");

    // Return the job ID from the response
    return id;
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
    auto samplesJson = getJobResponse["results"];
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


  std::string extractOutputLog(ServerMessage &postJobResponse,
                               std::string &jobId) override {
    CUDAQ_INFO("extractOutputLog: {}, {}", jobId, postJobResponse.dump());
    //TODO: implement
    return postJobResponse["results"];
  }

  void updatePassPipeline(
  const std::filesystem::path &platformPath, std::string &passPipeline) override {
    CUDAQ_INFO("updatePassPipeline: platformPath: {}, passPipeline: {}", platformPath.string(), passPipeline);
    std::string mappingMode = backendConfig["qubit_mapping_mode"];
    if (mappingMode == "backend") {
      // Adding a simple "tail": do not run any qubit mapping.
      passPipeline += ",func.func(expand-control-veqs,combine-quantum-alloc,canonicalize,combine-measurements)";
      CUDAQ_INFO("After removing mapping pass, updated pass pipeline: {}", passPipeline);
      return;
    }
    std::filesystem::path qpuConfigPath = platformPath / "mapping/quantum_machines" / "latest_qpu_config.txt";
    std::string machineconfigFilePath = qpuConfigPath.string();
    if (mappingMode == "local_get_latest") {
      // If mapping is done locally with the latest qpu config from the backend, we need to get the latest qpu config file from the backend and provide that to the mapping pass.
      // Get the latest qpu config file from the backend and set quantumArchitectureFilePath to its path
      try {
        // Create a RestClient and get the latest qpu config from backendConfig["url"]+"/v1/config/qubits" from the backend
        // Store the response in a file in the platformPath / "mapping/quantum_machines" directory, and set quantumArchitectureFilePath to that file path
        RestClient client;
        client.setVerbose(true); 
        auto headers = getHeaders();
        auto response = client.getRawText(backendConfig["url"], "/v1/config/qubits", headers);
        std::string qpuConfig = response;
        CUDAQ_INFO("Updated configuration: {}", qpuConfig);
        std::filesystem::create_directories(qpuConfigPath.parent_path());
        std::ofstream outFile(qpuConfigPath);
        outFile << qpuConfig;
        outFile.close();

      } catch (const std::exception &e) {
        throw std::runtime_error("Failed to get latest qpu config from backend: " +
                                  std::string(e.what()));
        }
    } 
    else if (mappingMode != "local_file") {
      throw std::runtime_error("qubit_mapping_mode: " + mappingMode + " is not supported. Supported modes are 'local-file', 'local-get-latest', and 'backend'.");
    }
    // Add the pipelines that ar responsible for qubit mapping, and adjust he file path
    passPipeline += ",func.func(expand-control-veqs,add-dealloc,combine-quantum-alloc,canonicalize,factor-quantum-alloc,memtoreg),add-wireset,func.func(assign-wire-indices),qubit-mapping{device=file(%QPU_ARCH%)},func.func(regtomem)";
    passPipeline =
        std::regex_replace(passPipeline, std::regex("%QPU_ARCH%"), machineconfigFilePath);
    CUDAQ_INFO("Updated pass pipeline: {}", passPipeline);
  }
};

} // namespace cudaq

// Register the Quantum Machines server helper in the CUDA-Q server helper
// factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantumMachinesServerHelper,
                    quantum_machines)
