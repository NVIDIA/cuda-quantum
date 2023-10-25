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

#include "nlohmann/json.hpp"

#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace cudaq {

class IQMServerHelper : public ServerHelper {
protected:
  /// @brief The base URL
  std::string iqmServerUrl = "http://localhost/cocos/";

  /// @brief IQM QPU architecture, provided during the compile-time
  std::string qpuArchitecture = "Adonis";

  /// @brief The default cortex-cli tokens file path
  std::optional<std::string> tokensFilePath = std::nullopt;

  /// @brief Program target QPU architecture
  std::string targetArchitecture = "";

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Parse cortex-cli tokens JSON for the API access token
  std::optional<std::string> readApiToken() const {
    if (!tokensFilePath.has_value()) {
      cudaq::info(
          "tokensFilePath is not set, assuming no authentication is required");
      return std::nullopt;
    }

    std::string unwrappedTokensFilePath = tokensFilePath.value();
    std::ifstream tokensFile(unwrappedTokensFilePath);
    if (!tokensFile.is_open()) {
      throw std::runtime_error("Unable to open tokens file: " +
                               unwrappedTokensFilePath);
    }
    nlohmann::json tokens;
    tokensFile >> tokens;
    tokensFile.close();
    return tokens["access_token"].get<std::string>();
  }

  /// @brief Get server quantum architecture name
  std::string getQuantumArchitectureName() const {
    RestClient client;
    auto headers = generateRequestHeader();
    auto quantumArchitecture =
        client.get(iqmServerUrl, "quantum-architecture", headers);
    try {
      cudaq::debug("quantumArchitecture = {}", quantumArchitecture.dump());
      return quantumArchitecture["quantum_architecture"]["name"]
          .get<std::string>();
    } catch (const std::exception &e) {
      throw std::runtime_error("Unable to get quantum architecture name: " +
                               std::string(e.what()));
    }
  }

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "iqm"; }
  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    bool emulate = false;
    auto iter = backendConfig.find("emulate");
    if (iter != backendConfig.end()) {
      emulate = iter->second == "true";
    }

    // Set QPU architecture
    iter = backendConfig.find("qpu-architecture");
    if (iter == backendConfig.end()) {
      throw std::runtime_error("QPU architecture is not provided");
    }
    qpuArchitecture = iter->second;
    cudaq::debug("qpuArchitecture = {}", qpuArchitecture);

    // Set an alternate base URL if provided.
    iter = backendConfig.find("url");
    if (iter != backendConfig.end()) {
      iqmServerUrl = iter->second;
    }

    // Allow overriding IQM Server Url, the compiled program will still work if
    // architecture matches. This is useful in case we're using the same program
    // against different backends, for example simulated and actually connected
    // to the hardware.
    auto envIqmServerUrl = getenv("IQM_SERVER_URL");
    if (envIqmServerUrl) {
      iqmServerUrl = std::string(envIqmServerUrl);
    }

    if (!iqmServerUrl.ends_with("/"))
      iqmServerUrl += "/";
    cudaq::debug("iqmServerUrl = {}", iqmServerUrl);

    if (emulate) {
      cudaq::info(
          "Emulation is enabled, ignore tokens file and IQM Server URL");
      return;
    }

    // Set alternative cortex-cli tokens file path if provided via env var
    auto envTokenFilePath = getenv("IQM_TOKENS_FILE");
    auto defaultTokensFilePath =
        std::string(getenv("HOME")) + "/.cache/iqm-cortex-cli/tokens.json";
    cudaq::debug("defaultTokensFilePath = {}", defaultTokensFilePath);
    if (envTokenFilePath) {
      tokensFilePath = std::string(envTokenFilePath);
    } else if (cudaq::fileExists(defaultTokensFilePath)) {
      tokensFilePath = defaultTokensFilePath;
    }
    cudaq::debug("tokensFilePath = {}", tokensFilePath.value_or("not set"));

    // Fetch quantum-architecture program was compiled with
    auto configuredTargetArchitecture = getQuantumArchitectureName();
    cudaq::debug("configuredTargetArchitecture = {}",
                 configuredTargetArchitecture);

    // Does it match the compiled architecture?
    if (qpuArchitecture != configuredTargetArchitecture) {
      throw std::runtime_error(
          "IQM QPU architecture mismatch: " + qpuArchitecture +
          " != " + configuredTargetArchitecture);
    }
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

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;
};

ServerJobPayload
IQMServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;

  // cuda-quantum expects every circuit to be a separate job,
  // so we cannot use the batch mode
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message = ServerMessage::object();
    message["circuits"] = ServerMessage::array();
    message["shots"] = shots;

    ServerMessage yac = nlohmann::json::parse(circuitCode.code);
    yac["name"] = circuitCode.name;
    message["circuits"].push_back(yac);
    messages.push_back(message);
  }

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  // return the payload
  return std::make_tuple(iqmServerUrl + "jobs", headers, messages);
}

std::string IQMServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string IQMServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return "jobs" + postResponse["id"].get<std::string>() + "/counts";
}

std::string IQMServerHelper::constructGetJobPath(std::string &jobId) {
  return iqmServerUrl + "jobs/" + jobId + "/counts";
}

bool IQMServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  cudaq::debug("getJobResponse: {}", getJobResponse.dump());

  auto jobStatus = getJobResponse["status"].get<std::string>();
  std::unordered_set<std::string> terminalStatuses = {"ready", "failed",
                                                      "aborted"};
  return terminalStatuses.find(jobStatus) != terminalStatuses.end();
}

cudaq::sample_result
IQMServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobID) {
  cudaq::info("postJobResponse: {}", postJobResponse.dump());

  // check if the job succeeded
  auto jobStatus = postJobResponse["status"].get<std::string>();
  if (jobStatus != "ready") {
    auto jobMessage = postJobResponse["message"].get<std::string>();
    throw std::runtime_error("Job status: " + jobStatus +
                             ", reason: " + jobMessage);
  }

  auto counts_batch = postJobResponse["counts_batch"];
  if (counts_batch.is_null()) {
    throw std::runtime_error("No counts in the response");
  }

  // assume there is only one measurement and everything goes into the
  // GlobalRegisterName of `sample_results`
  std::vector<ExecutionResult> srs;

  for (auto &counts : counts_batch.get<std::vector<ServerMessage>>()) {
    srs.push_back(ExecutionResult(
        counts["counts"].get<std::unordered_map<std::string, std::size_t>>()));
  }

  return sample_result(srs);
}

std::map<std::string, std::string>
IQMServerHelper::generateRequestHeader() const {
  std::map<std::string, std::string> headers{
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"User-Agent", "cudaq/IQMServerHelper"},
      {"Accept", "*/*"}};
  auto apiToken = readApiToken();
  if (apiToken.has_value()) {
    headers["Authorization"] = "Bearer " + apiToken.value();
  };
  return headers;
}

RestHeaders IQMServerHelper::getHeaders() { return generateRequestHeader(); }

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IQMServerHelper, iqm)
