/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors. *
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
#include <bitset>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_set>

using json = nlohmann::json;

namespace cudaq {
std::string lowercaseArgument(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

bool booleanArgument(const std::string &string_argument) {
  return lowercaseArgument(string_argument) ==
         "true"; // we should handle wrong string-boolean values
}

// Retrieve an environment variable
std::string getEnvVar(const std::string &key, const std::string &defaultVal,
                      const bool isRequired) {
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, either return the default or throw an
  // exception
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error(key + " environment variable is not set.");
    else
      return defaultVal;
  }
  return std::string(env_var);
}

/// @brief The TiiServerHelper class extends the ServerHelper class
/// to handle interactions with the TII server for submitting and
/// retrieving quantum computation jobs.
class TiiServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://q-cloud.tii.ae";
  static constexpr const char *DEFAULT_VERSION = "0.2.2";

public:
  const std::string name() const override { return "tii"; }

  /// @brief Return POST/GET headers required for the TII server.
  RestHeaders getHeaders() override {
    RestHeaders headers;
    headers["Content-Type"] = "application/json";

    // If Authentication token is not provided explicitly, it is read from the
    // `TII_API_TOKEN` environment variable.
    if (backendConfig.count("api_key"))
      headers["x-api-token"] = backendConfig["api_key"];
    else
      headers["x-api-token"] = getEnvVar("TII_API_TOKEN", "", true);

    headers["x-qibo-client-version"] = backendConfig["version"];

    return headers;
  }

  /// @brief Initialize the TII server with the provided configuration.
  void initialize(BackendConfig config) override {
    CUDAQ_INFO("Initializing TII Backend");
    backendConfig = config;

    if (!backendConfig.count("url"))
      backendConfig["url"] = getEnvVar("TII_API_URL", DEFAULT_URL, false);
    auto tii_url = backendConfig["url"];
    // append a trailing slash to complete the path later
    backendConfig["url"] = tii_url.ends_with("/") ? tii_url : tii_url + "/";
    if (!backendConfig.count("version"))
      backendConfig["version"] = DEFAULT_VERSION;
    if (!backendConfig.count("verbatim"))
      backendConfig["verbatim"] = "false";

    // Set shots if provided
    if (config.find("shots") != config.end())
      this->setShots(std::stoul(config["shots"]));
  }

  /// @brief Create and return the job payload given the compiled quantum
  /// circuit code for submission
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    ServerMessage job;

    job["circuit"] = circuitCodes[0].code;
    job["nshots"] = shots;
    job["device"] = backendConfig["device"];
    job["project"] = backendConfig["project"];
    job["verbatim"] = booleanArgument(backendConfig["verbatim"]);

    RestHeaders headers = getHeaders();
    std::string path = "api/jobs";

    return std::make_tuple(backendConfig["url"] + path, headers,
                           std::vector<ServerMessage>{job});
  }

  /// @brief Extract job id from the GET returned by the TII server.
  std::string extractJobId(ServerMessage &postResponse) override {
    if (!postResponse.contains("pid"))
      return "";

    return postResponse.at("pid");
  }

  /// @brief Track job id on the TII server.
  std::string constructGetJobPath(ServerMessage &postResponse) override {
    return extractJobId(postResponse);
  }

  /// @brief Generate full url for tracking the job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    return backendConfig["url"] + "api/jobs/" + jobId;
  }

  /// @brief Control the status of the job.
  /// Return true if the job succeeds or fails.
  bool jobIsDone(ServerMessage &getJobResponse) override {
    if (!getJobResponse.contains("status"))
      return false;

    std::string status = getJobResponse["status"];
    return status == "success" || status == "error";
  }

  /// @brief Example implementation of result processing.
  ///
  /// The raw results from quantum hardware often need post-processing (bit
  /// reordering, normalization, etc.) to match CUDA-Q's expectations.
  /// This is the place to do that.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override {
    CUDAQ_INFO("Processing results: {}", getJobResponse.dump());

    // Extract measurement results from the response
    auto samplesJson = getJobResponse["frequencies"];
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

  /// @brief Example implementation of polling configuration.
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(5);
  }
};

} // namespace cudaq

// Register the server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::TiiServerHelper, tii)
