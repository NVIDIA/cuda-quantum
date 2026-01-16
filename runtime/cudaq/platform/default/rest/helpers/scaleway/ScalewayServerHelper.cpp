/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qio/Qio.h"
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
}

void
ScalewayServerHelper::ensureSessionIsActive() {
    if (!m_sessionId.empty()) {
    try {
      std::string response = performInternalRequest("GET", "/sessions/" + sessionId);
      auto j = json::parse(response);
      std::string status = j.value("status", "unknown");

      if (status == "error" || status == "stopped" || status == "stopping") {
        m_sessionId = "";
      } else {
        return;
      }
    } catch (...) {
      m_sessionId = "";
    }
  }

  if (m_sessionId.empty()) {
    m_platformName = getOption("machine", m_basePlatformName);
    m_projectId = getOption("project_id");

    json sessionPayload;
    sessionPayload["name"] = "cudaq-session-" + std::to_string(std::rand());
    sessionPayload["platform_id"] = m_platformName; // TODO: get platform id
    if (!m_projectId.empty()) {
      sessionPayload["project_id"] = m_projectId;
    }

    // Endpoint sans région
    std::string response = performInternalRequest("POST", "/sessions", sessionPayload);

    auto j = json::parse(response);
    if (j.contains("id")) {
      m_sessionId = j["id"].get<std::string>();
    } else {
      throw std::runtime_error("Failed to create session: " + response);
    }
  }
}

std::string
ScalewayServerHelper::serializeKernelToQio(const std::string& code) {
    qio::QQuantumProgram program(
        kernel,
        qio::SerializationFormat::QIR,
        qio::CompressionFormat::GZIP);

    qio::QuantumComputationParameters params(1024);

    qio::QioQuantumComputationModel model(program, params);

    return model.toJson().dump();
}

std::string
ScalewayServerHelper::createModel(const std::string& name, const std::string& content) {
    json circuitPayload;
    circuitPayload["name"] = name;
    circuitPayload["definition"]["openqasm"] = content;
    if (!m_projectId.empty()) {
        circuitPayload["project_id"] = m_projectId;
    }

    std::string response = m_client.post("/models", circuitPayload);

    auto j = json::parse(response);
    if (j.contains("id")) {
      return j["id"].get<std::string>();
    }

    throw std::runtime_error("Cannot upload kernel: " + response);
  }

virtual
std::map<std::string, std::string>
ScalewayServerHelper::getHeaders() override {
  std::map<std::string, std::string> headers;

  // put to initialize
  std::string apiKey = getOption("api_key");
  if (apiKey.empty()) {
      if (const char* envKey = std::getenv("SCW_SECRET_KEY")) apiKey = envKey;
  }

  headers["X-Auth-Token"] = apiKey;
  headers["Content-Type"] = "application/json";

  return headers;
}

ServerJobPayload
ScalewayServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) override {
  ServerJobPayload ret;
  std::vector<ServerMessage> &tasks = std::get<2>(ret);
  for (auto &circuitCode : circuitCodes) {
    ServerMessage taskRequest;
    taskRequest["name"] = circuitCode.name;
    taskRequest["session_id"] = m_sessionId;
    taskRequest["model_id"] = uploadModel(circuitCode.name, circuitCode.code);
    // taskRequest["sampling_count"] = shots;
    tasks.push_back(taskRequest);
  }
  CUDAQ_INFO("Created job payload for Scaleway, "
              "targeting platform {}",
              m_platformName);
  return ret;
}

virtual
std::string
ScalewayServerHelper::extractJobId(const std::string &postResponse) override {
  auto j = json::parse(postResponse);
  if (j.contains("id")) return j["id"].get<std::string>();
  throw std::runtime_error("Job submission failed: " + postResponse);
}

virtual
bool
ScalewayServerHelper::isJobCompleted(const std::string &getResponse) override {
  auto j = json::parse(getResponse);
  std::string status = j.value("status", "unknown");
  if (status == "error") {
      std::string err = j.contains("result") ? j["result"].value("error_message", "Unknown error") : "Unknown";
      throw std::runtime_error("Scaleway Job Error: " + err);
  }
  return (status == "completed" || status == "canceled");
}

sample_result
ScalewayServerHelper::processResults(ServerMessage &resultsJson,
                                                 std::string &jobID) {
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
      rawPayload = performInternalRequest("GET", downloadUrl, {}, true);
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

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ScalewayServerHelper, scaleway)