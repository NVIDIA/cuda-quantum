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
#include <regex>
#include <thread>

namespace cudaq {

/// @brief The QbraidServerHelper class extends the ServerHelper class to
/// handle interactions with the qBraid server for submitting and retrieving
/// quantum computation jobs to various qBraid supported devices.
class QbraidServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api-v2.qbraid.com/api/v1";
  static constexpr const char *DEFAULT_DEVICE = "qbraid:qbraid:sim:qir-sv";
  static constexpr int DEFAULT_QUBITS = 30;

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "qbraid"; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override {
    cudaq::info("Initializing qBraid Backend.");

    backendConfig.clear();
    backendConfig["url"] = getValueOrDefault(config, "url", DEFAULT_URL);
    backendConfig["user_agent"] = "cudaq/" + std::string(cudaq::getVersion());
    backendConfig["qubits"] = std::to_string(DEFAULT_QUBITS);

    // Accept "machine" as a user-friendly alias for qBraid's device_id
    // Usage: cudaq.set_target("qbraid", machine="qbraid:qbraid:sim:qir-sv")
    if (!config["machine"].empty()) {
      backendConfig["device_id"] = config["machine"];
    } else {
      backendConfig["device_id"] =
          getValueOrDefault(config, "device_id", DEFAULT_DEVICE);
    }

    // Accept api_key from target arguments, fall back to QBRAID_API_KEY env var
    // Usage: cudaq.set_target("qbraid", api_key="my-key")
    bool isApiKeyRequired = [&]() {
      auto it = config.find("emulate");
      if (it != config.end() && it->second == "true")
        return false;
      return true;
    }();
    if (!config["api_key"].empty()) {
      backendConfig["api_key"] = config["api_key"];
    } else {
      backendConfig["api_key"] =
          getEnvVar("QBRAID_API_KEY", "", isApiKeyRequired);
    }
    backendConfig["job_path"] = backendConfig["url"] + "/jobs";

    if (!config["shots"].empty()) {
      backendConfig["shots"] = config["shots"];
      this->setShots(std::stoul(config["shots"]));
    } else {
      backendConfig["shots"] = "1000";
      this->setShots(1000);
    }

    parseConfigForCommonParams(config);

    cudaq::info("qBraid configuration initialized:");
    for (const auto &[key, value] : backendConfig) {
      if (key == "api_key") {
        cudaq::info("  api_key = <redacted, {} chars>", value.size());
      } else {
        cudaq::info("  {} = {}", key, value);
      }
    }
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    if (backendConfig.find("job_path") == backendConfig.end()) {
      throw std::runtime_error(
          "job_path not found in config. Was initialize() called?");
    }

    std::vector<ServerMessage> jobs;
    for (auto &circuitCode : circuitCodes) {
      ServerMessage job;
      job["deviceQrn"] = backendConfig.at("device_id");
      // Use the per-call shots (set via cudaq::sample(..., shots_count=N))
      job["shots"] = shots;

      // v2 API: program is a structured object with format and data
      nlohmann::json program;
      program["format"] = "qasm2";
      program["data"] = circuitCode.code;
      job["program"] = program;

      // v2 API: name is a top-level field (not nested under tags)
      if (!circuitCode.name.empty()) {
        job["name"] = circuitCode.name;
      }

      jobs.push_back(job);
    }

    return std::make_tuple(backendConfig.at("job_path"), getHeaders(), jobs);
  }

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override {
    // v2 API: jobQrn is nested under data envelope
    if (postResponse.contains("data") &&
        postResponse["data"].contains("jobQrn")) {
      return postResponse["data"]["jobQrn"].get<std::string>();
    }
    throw std::runtime_error(
        "ServerMessage doesn't contain 'data.jobQrn' key.");
  }

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // v2 API: use path parameter instead of query parameter
    if (postResponse.contains("data") &&
        postResponse["data"].contains("jobQrn")) {
      return backendConfig.at("job_path") + "/" +
             postResponse["data"]["jobQrn"].get<std::string>();
    }
    throw std::runtime_error(
        "ServerMessage doesn't contain 'data.jobQrn' key.");
  }

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    // v2 API: /jobs/{jobQrn}
    return backendConfig.at("job_path") + "/" + jobId;
  }

  /// @brief Constructs the URL for retrieving the measurement results of a
  /// completed job based on a job ID.
  std::string constructGetResultsPath(const std::string &jobId) {
    // v2 API: /jobs/{jobQrn}/result
    return backendConfig.at("job_path") + "/" + jobId + "/result";
  }

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override {
    std::string status;

    // v2 API: status is nested under data envelope
    if (getJobResponse.contains("data") &&
        getJobResponse["data"].contains("status")) {
      status = getJobResponse["data"]["status"].get<std::string>();
      cudaq::info("Job status from v2 data envelope: {}", status);
    } else if (getJobResponse.contains("status")) {
      // Fallback: direct status field
      status = getJobResponse["status"].get<std::string>();
      cudaq::info("Job status from direct response: {}", status);
    } else {
      cudaq::info("Unexpected job response format: {}", getJobResponse.dump());
      throw std::runtime_error("Invalid job response format");
    }

    if (status == "FAILED" || status == "COMPLETED" || status == "CANCELLED") {
      return true;
    }

    return false;
  }

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override {
    // qbraid's v2 API has a window where status transitions to COMPLETED
    // before the result payload is queryable on /result, so /result returns
    // {success: false, data: {message: "not yet available"}}. Retry with
    // backoff absorbs that race. Exercised deterministically via the mock's
    // POST /test/delay_next_results endpoint (see checkResultRetry /
    // checkResultRetryExhaustion tests).
    const int maxRetries = 3;
    const int waitTime = 2;
    const float backoffFactor = 2.0;

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
      try {
        auto resultsPath = constructGetResultsPath(jobId);
        auto headers = getHeaders();

        cudaq::info("Fetching results from v2 endpoint (attempt {}/{}): {}",
                    attempt + 1, maxRetries, resultsPath);
        RestClient client;
        auto resultJson = client.get("", resultsPath, headers, true);

        // v2 API: error indicated by success=false
        if (resultJson.contains("success") &&
            resultJson["success"].is_boolean() &&
            !resultJson["success"].get<bool>()) {
          std::string errorMsg = "Results not yet available";
          if (resultJson.contains("data") &&
              resultJson["data"].contains("message")) {
            errorMsg = resultJson["data"]["message"].get<std::string>();
          }
          cudaq::info("Results endpoint returned success=false: {}", errorMsg);

          if (attempt == maxRetries - 1) {
            throw std::runtime_error("Error retrieving results: " + errorMsg);
          }
        }
        // v2 API: measurementCounts nested under data.resultData
        else if (resultJson.contains("data") &&
                 resultJson["data"].contains("resultData") &&
                 resultJson["data"]["resultData"].contains(
                     "measurementCounts")) {
          cudaq::info("Processing results from v2 endpoint");
          CountsDictionary counts;
          auto &measurements =
              resultJson["data"]["resultData"]["measurementCounts"];

          for (const auto &[bitstring, count] : measurements.items()) {
            counts[bitstring] =
                count.is_number()
                    ? static_cast<std::size_t>(count.get<double>())
                    : static_cast<std::size_t>(count);
          }

          // The returned bitstring spans every measured qubit, including
          // compiler-generated ancillae that the user never declared. Reduce
          // it down to the user-visible qubits using the output_names entry
          // populated by the framework (Executor.cpp writes one per submitted
          // circuit; Future.cpp re-initializes the helper with that config
          // before processResults runs). Mirrors the IonQ / Braket helpers.
          cudaq::ExecutionResult fullExecResults{counts};
          auto fullSampleResults = cudaq::sample_result{fullExecResults};

          std::vector<ExecutionResult> execResults;

          auto outputNamesIt = outputNames.find(jobId);
          if (outputNamesIt != outputNames.end() &&
              !outputNamesIt->second.empty()) {
            auto &job_output_names = outputNamesIt->second;

            std::vector<std::size_t> qubitNumbers;
            qubitNumbers.reserve(job_output_names.size());
            for (auto &[result, info] : job_output_names)
              qubitNumbers.push_back(info.qubitNum);

            auto subset = fullSampleResults.get_marginal(qubitNumbers);
            execResults.emplace_back(ExecutionResult{subset.to_map()});

            // Emit one single-bit register per named result so that
            // `sample_result::to_map(registerName)` still works.
            for (const auto &[result, info] : job_output_names) {
              CountsDictionary regCounts;
              for (const auto &[bits, count] : fullSampleResults)
                regCounts[std::string{bits[info.qubitNum]}] += count;
              execResults.emplace_back(regCounts, info.registerName);
            }
          } else {
            // No output_names available: fall back to the full flat counts.
            execResults.emplace_back(ExecutionResult{counts});
          }

          return cudaq::sample_result(execResults);
        }

        // No valid data yet and no explicit error - retry
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0)
                              ? waitTime
                              : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("No valid results yet, retrying in {} seconds",
                      sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }

      } catch (const std::exception &e) {
        // RestClient throws std::runtime_error on any non-success HTTP status
        // (see runtime/common/RestClient.cpp) with a fixed message format:
        //   "HTTP <VERB> Error - status code <code>: <curl_err>: <body>"
        // The code isn't exposed as a structured attribute, so we parse it
        // out to distinguish terminal client errors (401/403/404/409) from
        // transient server/network errors (5xx, parse errors) that retry.
        static const std::regex statusRx(R"(status code (\d+))");
        const std::string what = e.what();
        std::smatch match;
        int statusCode = 0;
        if (std::regex_search(what, match, statusRx))
          statusCode = std::stoi(match[1]);

        // Terminal: auth failures - retrying will not recover.
        if (statusCode == 401 || statusCode == 403)
          throw std::runtime_error(
              "qBraid authentication failed (HTTP " +
              std::to_string(statusCode) +
              "). Verify QBRAID_API_KEY or api_key target argument.");

        // Terminal: result resource genuinely does not exist. This is
        // distinct from the "not yet available" race which returns
        // 200 + success=false (handled above).
        if (statusCode == 404)
          throw std::runtime_error(
              "qBraid result not found (HTTP 404) for job " + jobId +
              ". The job may have been deleted or never produced results.");

        // Terminal: job reached a non-success terminal state (FAILED or
        // CANCELLED). qBraid v2 returns 409 Conflict on /result in that case
        // because no measurement data will ever be produced.
        if (statusCode == 409)
          throw std::runtime_error(
              "qBraid job " + jobId +
              " did not produce results (HTTP 409). The job likely FAILED "
              "or was CANCELLED.");

        // Retryable: 5xx, network errors, JSON parse failures, etc.
        cudaq::info("Exception when fetching results (attempt {}/{}): {}",
                    attempt + 1, maxRetries, what);
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0)
                              ? waitTime
                              : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("Retrying in {} seconds", sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }
      }
    }

    throw std::runtime_error("Failed to retrieve measurement counts after " +
                             std::to_string(maxRetries) + " attempts");
  }

  /// @brief Override the polling interval method
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(1);
  }

private:
  /// @brief Returns the headers for the server requests.
  RestHeaders getHeaders() override {
    if (backendConfig.find("api_key") == backendConfig.end()) {
      throw std::runtime_error(
          "API key not found in config. Was initialize() called?");
    }

    RestHeaders headers;
    headers["X-API-KEY"] = backendConfig.at("api_key");
    headers["Content-Type"] = "application/json";
    headers["User-Agent"] = backendConfig.at("user_agent");
    return headers;
  }

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key, const std::string &defaultVal,
                        const bool isRequired) const {
    const char *env_var = std::getenv(key.c_str());
    if (env_var == nullptr) {
      if (isRequired) {
        throw std::runtime_error(key + " environment variable is not set.");
      }

      return defaultVal;
    }
    return std::string(env_var);
  }

  /// @brief Helper function to get a value from config or return a default
  /// value.
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const {
    return config.find(key) != config.end() ? config.at(key) : defaultValue;
  }
};
} // namespace cudaq

// Register the QbraidServerHelper with the name "qbraid" in the ServerHelper
// factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QbraidServerHelper, qbraid)
