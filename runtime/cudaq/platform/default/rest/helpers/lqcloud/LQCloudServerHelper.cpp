/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/runtime/logger/logger.h"

#include "nlohmann/json.hpp"
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace cudaq {

class LQCloudServerHelper final : public ServerHelper, public QirServerHelper {
  std::string serverUrl;
  std::string qpuName;
  std::string apiKey;
  std::string measurementType;
  // Entry of the configured QPU from GET /api/v1/qpus, fetched once and
  // shared by measurement-type resolution and qubit-mapping setup.
  nlohmann::json qpuInfo;
  std::string architectureFilePath;

  std::string safeErrorSummary(std::string message) const {
    if (!apiKey.empty())
      for (std::size_t pos = 0;
           (pos = message.find(apiKey, pos)) != std::string::npos; pos += 10)
        message.replace(pos, apiKey.size(), "[redacted]");
    message = message.substr(0, message.find_first_of("\r\n"));
    // Keep the explanation, not credentials, addresses or traceback paths.
    static const std::regex sensitive(
        R"(bearer\s+[^\s,;]+|)"
        R"((?:api[ _-]?key|x-api-key|token|password|secret|authorization))"
        R"(["']?\s*[:=]\s*(?:"[^"]*"|'[^']*'|[^\s,;]+)|)"
        R"([a-z][a-z0-9+.-]*://[^\s,;]+|[a-z]:[\\/][^\s,;]+|)"
        R"([\\/][^\s,;]+|\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?\b|)"
        R"(\b(?:[a-f0-9]{0,4}:){2,}[a-f0-9:]+|)"
        R"(\b[a-z0-9_-]+(?:\.[a-z0-9_-]+)*\.[a-z][a-z0-9_-]+\b|)"
        R"(\b(?:qk_|sk-|sk_|ghp_|glpat-|eyJ)[a-z0-9_.=-]+)",
        std::regex::icase);
    message = std::regex_replace(message, sensitive, "[redacted]");
    std::replace_if(
        message.begin(), message.end(),
        [](unsigned char c) { return c < 32 || c == 127; }, ' ');
    if (message.size() > 512)
      message = message.substr(0, 512) + "...";
    return message;
  }

  /// GET /api/v1/qpus and return the entry describing the configured QPU.
  const nlohmann::json &fetchQpuInfo() {
    if (!qpuInfo.is_null())
      return qpuInfo;
    RestClient client;
    client.setCredentialProtection(true);
    auto headers = getHeaders();
    auto qpus = client.get(serverUrl, "/api/v1/qpus", headers,
                           serverUrl.starts_with("https://"));
    for (const auto &qpu : qpus) {
      if (qpu.value("name", std::string()) == qpuName) {
        qpuInfo = qpu;
        return qpuInfo;
      }
    }
    throw std::runtime_error("LQCloud QPU is not listed by the server");
  }

  /// Pick the measurement variant for the configured QPU. Follows the
  /// platform's rule: prefer plain 'measure' when the QPU supports it, use
  /// 'measure2' (x21 + measure readout) when the QPU only supports that.
  void resolveMeasurementType(const std::string &requested) {
    if (requested == "measure" || requested == "measure2") {
      measurementType = requested;
      return;
    }
    if (requested != "auto")
      throw std::runtime_error(
          "LQCloud measurement_type must be measure, measure2, or auto");
    const auto &qpu = fetchQpuInfo();
    auto supported = qpu.find("supported_measurement_types");
    if (supported == qpu.end() || !supported->is_array())
      throw std::runtime_error("LQCloud QPU '" + qpuName +
                               "' did not report supported measurement types");

    bool hasPlain = false, hasTwoState = false;
    for (const auto &kind : *supported) {
      hasPlain |= kind == "measure";
      hasTwoState |= kind == "measure2";
    }
    if (hasPlain)
      measurementType = "measure";
    else if (hasTwoState)
      measurementType = "measure2";
    else
      throw std::runtime_error("LQCloud QPU '" + qpuName +
                               "' does not support measure or measure2");
  }

public:
  const std::string name() const override { return "lqcloud"; }

  ~LQCloudServerHelper() override {
    if (!architectureFilePath.empty() &&
        std::remove(architectureFilePath.c_str()) != 0)
      cudaq::info("LQCloud: failed to delete temporary QPU architecture "
                  "file '{}': {}",
                  architectureFilePath, std::strerror(errno));
  }

  void initialize(BackendConfig config) override {
    backendConfig = std::move(config);
    serverUrl = backendConfig.at("url");
    while (serverUrl.ends_with('/'))
      serverUrl.pop_back();
    qpuName = backendConfig.at("qpu");
    const char *key = std::getenv("LQCLOUD_API_KEY");
    if (!key)
      throw std::runtime_error(
          "LQCLOUD_API_KEY environment variable is not set");
    apiKey = key;
    auto requestedType = backendConfig.count("measurement_type")
                             ? backendConfig.at("measurement_type")
                             : "auto";
    resolveMeasurementType(requestedType);
    backendConfig["measurement_type"] = measurementType;
    if (backendConfig.count("shots"))
      setShots(std::stoul(backendConfig.at("shots")));
    parseConfigForCommonParams(backendConfig);
  }

  RestHeaders getHeaders() override {
    return {{"Accept", "application/json"},
            {"Content-Type", "application/json"},
            {"X-API-Key", apiKey}};
  }

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    std::vector<ServerMessage> messages;
    for (auto &execution : circuitCodes) {
      auto circuit = nlohmann::json::parse(execution.code);
      const auto &nativeInstructions = circuit.at("instructions");
      auto measurementCount = std::count_if(
          nativeInstructions.begin(), nativeInstructions.end(),
          [](const auto &op) { return op.at("name") == "measure"; });
      if (measurementType == "measure2") {
        nlohmann::json instructions = nlohmann::json::array();
        for (const auto &instruction : circuit.at("instructions")) {
          if (instruction.at("name") == "measure")
            instructions.push_back({{"name", "x21"},
                                    {"qubits", instruction.at("qubits")},
                                    {"clbits", nlohmann::json::array()},
                                    {"params", nlohmann::json::array()}});
          instructions.push_back(instruction);
        }
        circuit["instructions"] = std::move(instructions);
      }
      const auto returned = circuit.find("run_return_indices");
      const bool isRun = returned != circuit.end();
      if (isRun) {
        if (circuitCodes.size() != 1)
          throw std::runtime_error("LQCloud run requires one circuit");
        backendConfig["lqcloud.run_return_indices"] = returned->dump();
        backendConfig["lqcloud.run_width"] = circuit.at("n_clbits").dump();
        circuit.erase(returned);
      }
      circuit["shots"] = shots;
      circuit["result_format"] =
          isRun || measurementCount >= 12 ? "memory" : "counts";
      messages.push_back(
          {{"qpu_name", qpuName},
           {"command",
            {{"action", "run_circuit"}, {"circuit", std::move(circuit)}}}});
    }
    return std::make_tuple(serverUrl + "/api/v1/tasks/async", getHeaders(),
                           std::move(messages));
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    return postResponse.at("task_id").get<std::string>();
  }

  std::string constructGetJobPath(std::string &jobId) override {
    return serverUrl + "/api/v1/tasks/async/" + jobId;
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    auto jobId = extractJobId(postResponse);
    return constructGetJobPath(jobId);
  }

  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &) override {
    return std::chrono::seconds(1);
  }

  bool jobIsDone(ServerMessage &response) override {
    const auto status = response.at("status").get<std::string>();
    return status == "completed" || status == "failed" || status == "cancelled";
  }

  cudaq::sample_result processResults(ServerMessage &response,
                                      std::string &jobId) override {
    const auto status = response.at("status").get<std::string>();
    if (status != "completed") {
      const auto task =
          std::regex_match(jobId, std::regex("task_[0-9a-f]{16,32}"))
              ? jobId
              : "unknown";
      std::string type = "unknown";
      auto errorType = response.find("error_type");
      if (errorType != response.end() && errorType->is_string() &&
          (*errorType == "hardware_error" || *errorType == "validation_error" ||
           *errorType == "params_stale"))
        type = errorType->get<std::string>();
      auto error = response.find("error");
      auto summary = error != response.end() && error->is_string()
                         ? safeErrorSummary(error->get<std::string>())
                         : "No error details returned";
      throw std::runtime_error(
          "LQCloud task " + task + " " +
          (status == "failed" || status == "cancelled" ? status : "unknown") +
          " [" + type + "]: " + summary);
    }

    CountsDictionary counts;
    const auto &data = response.at("result");
    if (data.value("result_format", std::string()) == "memory") {
      for (const auto &bits : data.at("memory").get<std::vector<std::string>>())
        ++counts[bits];
    } else {
      for (const auto &[bits, count] : data.at("counts").items())
        counts[bits] = count.get<std::size_t>();
    }
    sample_result result{ExecutionResult{counts}};
    auto reorderIdxIt = reorderIdx.find(jobId);
    if (reorderIdxIt != reorderIdx.end()) {
      auto &order = reorderIdxIt->second;
      if (!order.empty())
        result.reorder(order);
    }
    return result;
  }

  std::string extractOutputLog(ServerMessage &response,
                               std::string &jobId) override {
    // Reuse the existing sanitized task failure diagnostic, without fabricating
    // a history from counts when the server did not return per-shot memory.
    if (response.at("status") != "completed")
      processResults(response, jobId);
    const auto &data = response.at("result");
    if (data.value("result_format", std::string()) != "memory")
      throw std::runtime_error("LQCloud run requires per-shot memory");
    const auto memory = data.at("memory").get<std::vector<std::string>>();
    const auto width = std::stoul(backendConfig.at("lqcloud.run_width"));
    const auto indices =
        nlohmann::json::parse(backendConfig.at("lqcloud.run_return_indices"))
            .get<std::vector<std::size_t>>();
    if (memory.size() != shots || indices.empty())
      throw std::runtime_error(
          "LQCloud run returned an invalid shot count or return list");
    for (auto index : indices)
      if (index >= width)
        throw std::runtime_error(
            "LQCloud run return index exceeds memory width");
    std::string log =
        "HEADER\tschema_id\tlabeled\nHEADER\tschema_version\t1.0\n";
    for (const auto &bits : memory) {
      if (bits.size() != width ||
          bits.find_first_not_of("01") != std::string::npos)
        throw std::runtime_error(
            "LQCloud run returned invalid measurement memory");
      log += "START\nOUTPUT\tARRAY\t" + std::to_string(indices.size()) +
             "\tarray<i1 x " + std::to_string(indices.size()) + ">\n";
      for (std::size_t i = 0; i < indices.size(); ++i)
        log += std::string("OUTPUT\tBOOL\t") + bits[indices[i]] + "\t[" +
               std::to_string(i) + "]\n";
      log += "END\t0\n";
    }
    return log;
  }

  /// Replace the %QPU_ARCH% placeholder in the pass pipeline with a device
  /// connectivity file for the configured QPU so the qubit-mapping pass can
  /// route arbitrary circuits onto the QPU topology. The file is normally
  /// generated on the fly from the platform's QPU listing; it can be
  /// overridden with the 'mapping_file' backend parameter or the
  /// LQCLOUD_QPU_ARCH environment variable (useful for testing).
  std::map<std::string, std::string>
  getPipelineSubstitutions(const std::filesystem::path &) override {
    std::string pathToFile;
    if (const char *filename = std::getenv("LQCLOUD_QPU_ARCH")) {
      pathToFile = filename;
    } else {
      auto iter = backendConfig.find("mapping_file");
      if (iter != backendConfig.end()) {
        pathToFile = iter->second;
      } else {
        pathToFile = writeQpuArchitectureFile();
      }
    }

    // Protect the path from shell globbing in parsePassPipeline.
    pathToFile.insert(0, "'").append("'");
    return {{"%QPU_ARCH%", pathToFile}};
  }

private:
  /// Build the device connectivity file (Graph dump format understood by
  /// cudaq::Device::file) from the coupling map published by the platform.
  std::string writeQpuArchitectureFile() {
    if (!architectureFilePath.empty())
      return architectureFilePath;

    auto topology = fetchQpuInfo().at("topology");
    unsigned numQubits = fetchQpuInfo().value("qubits", 0);
    std::map<unsigned, std::set<unsigned>> adjacency;
    for (const auto &edge : topology.at("coupling_map")) {
      auto a = edge.at(0).get<unsigned>();
      auto b = edge.at(1).get<unsigned>();
      adjacency[a].insert(b);
      adjacency[b].insert(a);
      numQubits = std::max(numQubits, std::max(a, b) + 1);
    }
    if (numQubits == 0 || adjacency.empty())
      throw std::runtime_error("LQCloud QPU '" + qpuName +
                               "' did not report a usable topology");

    architectureFilePath = std::string(P_tmpdir) + "/lqcloud-arch-XXXXXX";
    int fd = mkstemp(architectureFilePath.data());
    if (fd < 0) {
      auto failedPath = architectureFilePath;
      architectureFilePath.clear();
      throw std::runtime_error(
          "Cannot write LQCloud QPU architecture file: \"" + failedPath +
          "\" - " + std::string(std::strerror(errno)));
    }
    FILE *file = fdopen(fd, "w");
    if (!file) {
      const int errorCode = errno;
      auto failedPath = architectureFilePath;
      architectureFilePath.clear();
      close(fd);
      std::remove(failedPath.c_str());
      throw std::runtime_error(
          "Cannot write LQCloud QPU architecture file: \"" + failedPath +
          "\" - " + std::string(std::strerror(errorCode)));
    }

    fprintf(file, "# NOTE: automatically generated for LQCloud\n\n");
    fprintf(file, "Number of nodes: %u\n", numQubits);
    fprintf(file, "Number of edges: ?\n\n");
    for (unsigned qubit = 0; qubit < numQubits; ++qubit) {
      std::string line = std::to_string(qubit) + " --> {";
      bool first = true;
      for (unsigned neighbor : adjacency[qubit]) {
        if (!first)
          line += ", ";
        first = false;
        line += std::to_string(neighbor);
      }
      line += "}\n";
      fputs(line.c_str(), file);
    }
    fclose(file);
    return architectureFilePath;
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::LQCloudServerHelper, lqcloud)
