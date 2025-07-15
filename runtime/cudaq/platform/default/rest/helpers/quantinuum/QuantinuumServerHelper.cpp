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
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <iostream>
#include <thread>

namespace cudaq {

/// @brief The QuantinuumServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Quantinuum REST
/// server.
class QuantinuumServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://nexus.quantinuum.com/";
  /// @brief The machine we are targeting
  std::string machine = "H2-1SC";
  /// @brief Time string, when the last tokens were retrieved
  std::string timeStr = "";
  /// @brief The refresh token
  std::string refreshKey = "";
  /// @brief The API token for the remote server
  std::string apiKey = "";

  std::string userSpecifiedCredentials = "";
  std::string credentialsPath = "";

  /// @brief Quantinuum requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "quantinuum"; }

  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    // Set the machine
    auto iter = backendConfig.find("machine");
    if (iter != backendConfig.end())
      machine = iter->second;

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

  std::optional<std::string>
  getResultId(ServerMessage &getJobResponse) override;

  std::string constructGetResultPath(const std::string &resultId) override;

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) override;
};

ServerJobPayload
QuantinuumServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {

  std::vector<ServerMessage> messages;

  // Get the tokens we need

  refreshTokens();

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  // return the payload
  return std::make_tuple(baseUrl + "job", headers, messages);
}

std::string QuantinuumServerHelper::extractJobId(ServerMessage &postResponse) {
  // "job_id": "$response.body#/data.id"
  return postResponse["data"]["id"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + "jobs/v1beta3/" + extractJobId(postResponse);
}

std::string QuantinuumServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + "jobs/v1beta3/" + jobId;
}

bool QuantinuumServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Job status strings: "COMPLETED", "QUEUED", "SUBMITTED", "RUNNING",
  // "CANCELLED", "ERROR"
  const std::string jobStatus =
      getJobResponse["data"]["attributes"]["status"]["status"]
          .get<std::string>();
  if (jobStatus == "ERROR") {
    const std::string errorMsg =
        getJobResponse["data"]["attributes"]["status"]["error_detail"]
            .get<std::string>();
    throw std::runtime_error("Job failed with error: " + errorMsg);
  } else if (jobStatus == "CANCELLED") {
    throw std::runtime_error("Job was cancelled.");
  }
  return jobStatus == "COMPLETED";
}

std::optional<std::string>
QuantinuumServerHelper::getResultId(ServerMessage &getJobResponse) {
  if (!jobIsDone(getJobResponse)) {
    throw std::runtime_error("Job is not done, cannot retrieve result ID.");
  }
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
    throw std::runtime_error("No 'result_id' found in job response item.");
  }
  return item["result_id"].get<std::string>();
}

std::string
QuantinuumServerHelper::constructGetResultPath(const std::string &resultId) {
  assert(!resultId.empty() && "Result ID cannot be empty.");
  return baseUrl + "results/v1beta3/" + resultId;
}

cudaq::sample_result
QuantinuumServerHelper::processResults(ServerMessage &resultResponse,
                                       std::string &jobId) {

  auto results = resultResponse["data"]["attributes"]["counts"];

  CUDAQ_INFO("Results message: {}", results.dump());

  // TODO: handle register mapping and qubit numbers
  // This is just a very basic implementation that assumes all qubits are
  // measured.
  cudaq::CountsDictionary counts;

  for (const auto &element : results) {
    const auto bitValues = element["outcome"]["array"].get<std::vector<int>>();
    const auto count = element["count"].get<std::size_t>();
    // Convert bit values to a string representation
    std::string bits;
    for (const auto &bit : bitValues) {
      bits += std::to_string(bit);
    }
    counts[bits] = count;
  }

  return cudaq::sample_result{cudaq::ExecutionResult{counts}};
}

std::map<std::string, std::string>
QuantinuumServerHelper::generateRequestHeader() const {
  std::string apiKey, refreshKey, timeStr;
  // searchAPIKey(apiKey, refreshKey, timeStr, userSpecifiedCredentials);
  std::map<std::string, std::string> headers{
      {"Authorization", apiKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

RestHeaders QuantinuumServerHelper::getHeaders() {
  return generateRequestHeader();
}

/// Refresh the api key and refresh-token
void QuantinuumServerHelper::refreshTokens(bool force_refresh) {}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuantinuumServerHelper,
                    quantinuum)
