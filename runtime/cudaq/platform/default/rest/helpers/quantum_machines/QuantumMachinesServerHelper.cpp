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

/// @brief The QuantumMachinesServerHelper class extends the ServerHelper class to
/// handle interactions with the Quantum Machines server for submitting and retrieving
/// quantum computation jobs.
class QuantumMachinesServerHelper : public ServerHelper {
  // TODO: Replace with actual Quantum Machines API URL and version
  static constexpr const char *DEFAULT_URL = "https://api.quantum-machines.com";
  static constexpr const char *DEFAULT_VERSION = "v1.0.0";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "quantum_machines"; }

  /// @brief Returns the headers for the server requests.
  RestHeaders getHeaders() override {
    // TODO: Implement headers for Quantum Machines API
    RestHeaders headers;
    return headers;
  }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override {
    // TODO: Implement initialization for Quantum Machines
    cudaq::info("Initializing Quantum Machines Backend.");
    backendConfig = config;
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    // TODO: Implement job creation for Quantum Machines
    cudaq::info("In createJob");
    ServerMessage job;
    RestHeaders headers;
    return std::make_tuple("", headers, std::vector<ServerMessage>{job});
  }

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override {
    // TODO: Implement job ID extraction for Quantum Machines
    return "";
  }

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // TODO: Implement job path construction for Quantum Machines
    cudaq::info("In constructGetJobPath");
    return "";
  }

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    // TODO: Implement job path construction for Quantum Machines
    return "";
  }

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override {
    // TODO: Implement job status checking for Quantum Machines
    return false;
  }

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override {
    // TODO: Implement result processing for Quantum Machines
    cudaq::CountsDictionary counts;
    cudaq::ExecutionResult execResult{counts};
    return cudaq::sample_result{execResult};
  }

  /// @brief Override the polling interval method
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    // TODO: Implement polling interval for Quantum Machines
    return std::chrono::seconds(1);
  }
};

} // namespace cudaq

// Register the Quantum Machines server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantumMachinesServerHelper,
                    quantum_machines)
