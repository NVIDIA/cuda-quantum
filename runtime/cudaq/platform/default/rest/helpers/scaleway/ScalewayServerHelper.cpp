/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qio/Qio.h"
#include "qaas/Qaas.h"
#include "common/Logger.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <map>

using json = nlohmann::json;

namespace cudaq {

  void
  ScalewayServerHelper::initialize(BackendConfig config) {
      m_qaasClient.initialize(
          getOption("project_id"),
          getOption("secret_key"),
          getOption("url")
      );

      auto platformName = getOption("machine")

      m_platformName = if platformName.empty() ? m_defaultPlatformName : getOption("machine");
      m_sessionDeduplicationId = getOption("deduplicationId", "");
      m_sessionMaxDuration = getOption("maxDuration", "");
      m_sessionMaxIdleDuration = getOption("maxIdleDuration", "");
      m_sessionName = getOption("name", "cudaq-session");
  }

  std::chrono::microseconds
  ScalewayServerHelper::nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::microseconds(1000000);
  }

  bool
  ScalewayServerHelper::jobIsDone(ServerMessage &getJobResponse) override {
    auto j = json::parse(getJobResponse);
    std::string status = j.value("status", "unknown");
    if (status == "error") {
        std::string err = j.contains("result") ? j["result"].value("error_message", "Unknown error") : "Unknown";
        throw std::runtime_error("Scaleway Job Error: " + err);
    }
    return (status == "completed" || status == "canceled");
  }

  cudaq::sample_result
  ScalewayServerHelper::processResults(ServerMessage &postJobResponse,
                std::string &jobId) override {
  // Get results
  // For all result
  // Get raw result or URL
  // get payload
  // Unserialize to QuantumProgramResult
  // Convert list of result to CountsDictionary then ExecutionRsult
  // Finally return sample_result(ExecutionRsult[])

  auto j = json::parse(getResponse);

  if (!j.contains("job_results") || j["job_results"].empty()) {
      throw std::runtime_error("Job done but empty results.");
  }

  auto firstResult = j["job_results"][0];

  std::string rawPayload;

  if (firstResult.contains("result") && !firstResult["result"].is_null() && firstResult["result"] != "") {
      rawPayload = firstResult["result"].get<std::string>();
  }
  else if (firstResult.contains("url") && !firstResult["url"].is_null()) {
      std::string downloadUrl = firstResult["url"].get<std::string>();
      // rawPayload = performInternalRequest("GET", downloadUrl, {}, true);
  }
  else {
      throw std::runtime_error("invalid: empty 'result' and 'url' fields to get result.");
  }

  try {
      qio::QuantumProgramResult result = qio::QuantumProgramResult::fromJsonString(rawPayload);

      cudaq::CountsDictionary counts;

      for (const auto& sample : result.getSamples()) {
          std::string bitString;
          for (auto bit : sample.bits) {
              bitString += std::to_string(bit);
          }
          counts[bitString] += 1;
      }

      std::vector<ExecutionResult> execResults;
      execResults.emplace_back(ExecutionResult{counts});

      return cudaq::sample_result(execResults);

  } catch (const std::exception& e) {
    throw std::runtime_error("Error while parsing result: " + std::string(e.what()) + " | payload: " + rawPayload);
  }
}

  RestHeaders
  ScalewayServerHelper::getHeaders() override {
    return m_qaasClient.getHeader();
  }

  virtual ServerJobPayload
  ScalewayServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) override {
    ensureSessionIsActive();

    ServerJobPayload ret;
    std::vector<ServerMessage> &tasks = std::get<2>(ret);

    for (auto &circuitCode : circuitCodes) {
      ServerMessage taskRequest;
      std::string qioPayload = serializeKernelToQio(circuitCode.code);
      std::string modelId = m_qaasClient.createModel(qioPayload);
      taskRequest["model_id"] = modelId;
      taskRequest["session_id"] = m_sessionId;
      taskRequest["name"] = circuitCode.name;

      CUDAQ_INFO("Uploaded model to Scaleway with id {}", modelId);

      tasks.push_back(taskRequest);
    }

    CUDAQ_INFO("Created job payload for Scaleway, "
                "targeting platform {}",
                m_platformName);
    return ret;
  }

  std::string
  ScalewayServerHelper::extractJobId(ServerMessage &postResponse) override {
    auto j = json::parse(postResponse);
    if (j.contains("id")) return j["id"].get<std::string>();
    if (j.contains("job_id")) return j["job_id"].get<std::string>();
    throw std::runtime_error("Job submission failed: " + postResponse);
  }

  std::string
  ScalewayServerHelper::constructGetJobPath(std::string &jobId) override {
    return m_qaasClient.getJobResultsUrl(jobId);
  }

  std::string
  ScalewayServerHelper::constructGetJobPath(ServerMessage &postResponse) override {
    std::string jobId = extractJobId(postResponse);
    return m_qaasClient.getJobResultsUrl(jobId);
  }

  std::string
  ScalewayServerHelper::ensureSessionIsActive() {
    if (!m_sessionId.empty()) {
    try {
      auto session = m_qaasClient.getSession(m_sessionId);
      auto status = session.status;

      if (status == "error" || status == "stopped" || status == "stopping") {
        m_sessionId = "";
      } else {
        return m_sessionId;
      }
    } catch (...) {
      m_sessionId = "";
    }
  }

  if (m_sessionId.empty()) {
    m_platformName = getOption("machine", m_basePlatformName);

    auto platforms = m_qaasClient.listPlatforms(m_platformName);

    if (platforms.empty()) {
      throw std::runtime_error("No platforms found with name: " + m_platformName);
    }

    auto platform = platforms[0];

    CUDAQ_INFO("Creating session on Scaleway platform {} (id={})",
                platform.name, platform.id);

    auto session = m_qaasClient.createSession(
        platform.id,
        m_sessionName.empty() ? "cudaq-session-" + std::to_string(std::rand()) : m_sessionName,
        m_sessionDeduplicationId.empty() ? "" : m_sessionDeduplicationId,
        "", // No model id
        m_sessionMaxDuration.empty() ? "59m" : m_sessionMaxDuration,
        m_sessionMaxIdleDuration.empty() ? "59m" : m_sessionMaxIdleDuration,
        ""); // No parameters

    if (session.id.empty()) {
      throw std::runtime_error("Failed to create Scaleway session");
    }

    m_sessionId = session.id;
  }

  return m_sessionId;
}

  std::string
  ScalewayServerHelper::serializeKernelToQio(const std::string& code, size_t shots) {
      qio::QuantumProgram program(
          code,
          qio::SerializationFormat::QIR,
          qio::CompressionFormat::GZIP);

      qio::QuantumComputationParameters params(shots);

      qio::QuantumComputationModel model(program, params);

      return model.toJson().dump();
  }

// std::string
// ScalewayServerHelper::createModel(const std::string& payload) {
//     json circuitPayload;
//     circuitPayload["project_id"] = m_projectId;
//     circuitPayload["payload"] = payload;

//     std::string response = m_client.post("/models", circuitPayload);

//     auto j = json::parse(response);
//     if (j.contains("id")) {
//       return j["id"].get<std::string>();
//     }

//     throw std::runtime_error("Cannot upload kernel: " + response);
//   }

// virtual
// std::map<std::string, std::string>
// ScalewayServerHelper::getHeaders() override {
//   std::map<std::string, std::string> headers;
//   std::string apiKey = getOption("api_key");

//   if (apiKey.empty()) {
//       if (const char* envKey = std::getenv("SCW_SECRET_KEY")) apiKey = envKey;
//   }

//   headers["X-Auth-Token"] = apiKey;
//   headers["Content-Type"] = "application/json";

//   return headers;
// }

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ScalewayServerHelper, scaleway)