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
#include <map>
#include <regex>
#include <thread>

namespace cudaq {

class QbraidServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api-v2.qbraid.com/api/v1";
  static constexpr const char *DEFAULT_DEVICE = "qbraid:qbraid:sim:qir-sv";
  static constexpr int DEFAULT_QUBITS = 29;

public:
  const std::string name() const override { return "qbraid"; }

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
    if (!config["api_key"].empty()) {
      backendConfig["api_key"] = config["api_key"];
    } else {
      backendConfig["api_key"] = getEnvVar("QBRAID_API_KEY", "", true);
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
      cudaq::info("  {} = {}", key, value);
    }
  }

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
      program["data"] = normalizeClassicalRegisters(circuitCode.code);
      job["program"] = program;

      // v2 API: name is a top-level field (not nested under tags)
      if (!circuitCode.name.empty()) {
        job["name"] = circuitCode.name;
      }

      jobs.push_back(job);
    }

    return std::make_tuple(backendConfig.at("job_path"), getHeaders(), jobs);
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    // v2 API: jobQrn is nested under data envelope
    if (postResponse.contains("data") &&
        postResponse["data"].contains("jobQrn")) {
      return postResponse["data"]["jobQrn"].get<std::string>();
    }
    throw std::runtime_error(
        "ServerMessage doesn't contain 'data.jobQrn' key.");
  }

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

  std::string constructGetJobPath(std::string &jobId) override {
    // v2 API: /jobs/{jobQrn}
    return backendConfig.at("job_path") + "/" + jobId;
  }

  std::string constructGetResultsPath(const std::string &jobId) {
    // v2 API: /jobs/{jobQrn}/result
    return backendConfig.at("job_path") + "/" + jobId + "/result";
  }

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

  // Fetch results from v2 results endpoint with retry logic.
  //
  // Rationale: qbraid's v2 API has a window where status transitions to
  // COMPLETED before the result payload is queryable on /result, so /result
  // returns {success: false, data: {message: "not yet available"}}. The retry
  // with backoff absorbs that race.
  //
  // Exercised deterministically via the mock's POST /test/delay_next_results
  // endpoint (see checkResultRetry / checkResultRetryExhaustion tests).
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                      std::string &jobId) override {
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

          std::vector<ExecutionResult> execResults;
          execResults.emplace_back(ExecutionResult{counts});
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
        cudaq::info("Exception when fetching results: {}", e.what());
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
  // Merge multiple single-bit classical registers emitted by nvq++'s QASM 2
  // codegen into a single multi-bit `creg c[N]`. This is required to unblock
  // qBraid-routed hardware backends.
  //
  // Context: nvq++ emits one `creg varK[1];` per measurement. AWS Braket's
  // classical simulators (SV1, DM1, TN1) tolerate that via lenient register
  // concatenation, but stricter hardware transpilers below reject it:
  //   - IQM (Garnet etc.): returns only the first register -> 1-bit results
  //   - Rigetti: collapses all registers onto b[0] -> "bit already in use"
  //   - IonQ-via-Braket: similar strict behavior
  // Normalizing to a single register is the canonical QASM 2 form and is
  // accepted uniformly by every qBraid-reachable backend.
  std::string normalizeClassicalRegisters(const std::string &qasm) const {
    static const std::regex cregDeclRx(R"(creg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;)");

    std::vector<std::pair<std::string, int>> cregs;
    for (auto it = std::sregex_iterator(qasm.begin(), qasm.end(), cregDeclRx);
         it != std::sregex_iterator(); ++it) {
      cregs.emplace_back((*it)[1].str(), std::stoi((*it)[2].str()));
    }

    // Nothing to do if the QASM already has a single classical register.
    if (cregs.size() <= 1)
      return qasm;

    std::map<std::string, int> offsetByName;
    int totalBits = 0;
    for (auto &[name, size] : cregs) {
      offsetByName[name] = totalBits;
      totalBits += size;
    }

    std::string out = qasm;

    // Rewrite every `-> NAME[i]` target BEFORE we mutate the creg declarations.
    for (auto &[name, size] : cregs) {
      int base = offsetByName[name];
      for (int i = 0; i < size; ++i) {
        std::regex measureTargetRx("->\\s*" + name + "\\s*\\[\\s*" +
                                   std::to_string(i) + "\\s*\\]");
        out = std::regex_replace(out, measureTargetRx,
                                 "-> qbraid__creg__[" +
                                     std::to_string(base + i) + "]");
      }
    }

    // Replace the first declaration with the merged register.
    out = std::regex_replace(out, cregDeclRx,
                             "creg qbraid__creg__[" +
                                 std::to_string(totalBits) + "];",
                             std::regex_constants::format_first_only);

    // Remove the remaining original declarations.
    for (size_t i = 1; i < cregs.size(); ++i) {
      std::regex toRemove("creg\\s+" + cregs[i].first +
                          "\\s*\\[\\s*\\d+\\s*\\]\\s*;\\s*");
      out = std::regex_replace(out, toRemove, "");
    }

    cudaq::info(
        "Normalized {} classical registers into single qbraid__creg__[{}]",
        cregs.size(), totalBits);
    return out;
  }

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

  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const {
    return config.find(key) != config.end() ? config.at(key) : defaultValue;
  }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QbraidServerHelper, qbraid)
