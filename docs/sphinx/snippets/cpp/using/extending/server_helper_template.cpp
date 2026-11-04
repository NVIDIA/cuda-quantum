// ============================================================================ //
// Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   //
// All rights reserved.                                                         //
//                                                                              //
// This source code and the accompanying materials are made available under     //
// the terms of the Apache License 2.0 which accompanies this distribution.     //
// ============================================================================ //

// ServerHelper Template for REST-Style Backend
// This is a complete template for implementing a ServerHelper subclass.

#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"

namespace cudaq {

/// @brief Handles job submission and result retrieval for Provider Name.
class ProviderNameServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api.provider-name.com";

public:
  const std::string name() const override { return "<provider_name>"; }

  void initialize(BackendConfig config) override {
    backendConfig = config;
    parseConfigForCommonParams(backendConfig);
    if (!backendConfig.count("url"))
      backendConfig["url"] = DEFAULT_URL;
    if (auto it = config.find("shots"); it != config.end())
      setShots(std::stoul(it->second));
  }

  RestHeaders getHeaders() override {
    RestHeaders headers;
    headers["Content-Type"] = "application/json";
    if (backendConfig.count("api_key"))
      headers["Authorization"] = "Bearer " + backendConfig["api_key"];
    return headers;
  }

  /// @brief Build one task JSON per compiled kernel and POST them together.
  ServerJobPayload createJob(std::vector<KernelExecution> &circuitCodes) override {
    std::vector<ServerMessage> tasks;
    tasks.reserve(circuitCodes.size());
    for (const auto &circuit : circuitCodes) {
      ServerMessage task;
      task["content"] = circuit.code;
      task["shots"] = shots;
      tasks.push_back(std::move(task));
    }
    return {backendConfig["url"] + "/jobs", getHeaders(), std::move(tasks)};
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    if (!postResponse.contains("id"))
      return "";
    return postResponse.at("id");
  }

  /// @brief Both overloads must return a full URL, not just the job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    return backendConfig["url"] + "/jobs/" + jobId;
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    auto jobId = extractJobId(postResponse);
    return constructGetJobPath(jobId);
  }

  bool jobIsDone(ServerMessage &getJobResponse) override {
    if (!getJobResponse.contains("status"))
      return false;
    std::string status = getJobResponse["status"];
    return status == "COMPLETED" || status == "FAILED";
  }

  /// @brief Map provider result counts to a CUDA-Q sample_result.
  ///
  /// Raw results from quantum hardware often need post-processing (bit
  /// reordering, normalization, etc.) to match CUDA-Q's expectations.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                       std::string &jobId) override {
    auto samplesJson = postJobResponse["results"]["counts"];
    cudaq::CountsDictionary counts;
    for (auto &[bitstring, count] : samplesJson.items())
      counts[bitstring] = count;
    return cudaq::sample_result{cudaq::ExecutionResult{counts}};
  }

  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(5);
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ProviderNameServerHelper, <provider_name>)