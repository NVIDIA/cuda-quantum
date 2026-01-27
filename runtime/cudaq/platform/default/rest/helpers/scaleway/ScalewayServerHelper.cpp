/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "ScalewayServerHelper.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <map>
#include <regex>
#include <sstream>

using json = nlohmann::json;
using namespace cudaq; 

// Helper function to get a value from config or return a default
std::string getValueOrDefault(const BackendConfig &config,
                              const std::string &key,
                              const std::string &envKey,
                              const std::string &defaultValue) {
  auto it = config.find(key);
  auto envValue = !envKey.empty() ? std::string(std::getenv(envKey.c_str())) : "";
  auto providedValue = (it != config.end()) ? it->second : envValue;

  return !providedValue.empty() ? providedValue : defaultValue;
}

std::string serializeParametersToQio(size_t nb_shots) {
  qio::QuantumComputationParameters parameters(nb_shots, {});

  return parameters.toJson().dump();
}

std::string serializeKernelToQio(const std::string &code) {
  qio::QuantumProgram program(code, qio::QuantumProgramSerializationFormat::QASM_V2,
                              qio::CompressionFormat::ZLIB_BASE64_V1);

  std::vector<qio::QuantumProgram> programs = {program};

  qio::QuantumComputationModel model(programs);

  return model.toJson().dump();
}

void ScalewayServerHelper::initialize(BackendConfig config) {
  backendConfig = config;
  m_qaasClient = std::make_unique<qaas::v1alpha1::V1Alpha1Client>(
                      getValueOrDefault(config, "project_id", "SCW_PROJECT_ID", ""),
                      getValueOrDefault(config, "secret_key", "SCW_SECRET_KEY", ""),
                      getValueOrDefault(config, "url", "SCW_API_URL", ""));

  m_targetPlatformName = getValueOrDefault(config, "machine", "", DEFAULT_PLATFORM_NAME);
  m_sessionMaxDuration = getValueOrDefault(config, "max_duration", "", DEFAULT_MAX_DURATION);
  m_sessionMaxIdleDuration = getValueOrDefault(config, "max_idle_duration", "", DEFAULT_MAX_IDLE_DURATION);
  m_sessionDeduplicationId = getValueOrDefault(config, "deduplication_id", "", "");
  m_sessionName = getValueOrDefault(config, "name", "", "cudaq-session-" + std::to_string(std::rand()));

  setShots(std::stoul(getValueOrDefault(config, "shots", "", "1000")));
}

RestHeaders ScalewayServerHelper::getHeaders() {
  return m_qaasClient->getHeaders();
}

ServerJobPayload ScalewayServerHelper::createJob(
    std::vector<KernelExecution> &circuitCodes) {
  ensureSessionIsActive();

  ServerJobPayload ret;
  std::vector<ServerMessage> tasks;
  auto headers = m_qaasClient->getHeaders();

  for (auto &circuitCode : circuitCodes) {
    ServerMessage taskRequest;
    std::string qioPayload = serializeKernelToQio(circuitCode.code);
    std::string qioParams = serializeParametersToQio(shots);
    qaas::v1alpha1::Model model = m_qaasClient->createModel(qioPayload);

    taskRequest["model_id"] = model.id;
    taskRequest["session_id"] = m_sessionId;
    taskRequest["name"] = circuitCode.name;
    taskRequest["parameters"] = qioParams;

    tasks.push_back(taskRequest);
  }

  CUDAQ_INFO("Created job payload for Scaleway, "
             "targeting platform {}",
             m_targetPlatformName);

  return std::make_tuple(m_qaasClient->getJobsUrl(), headers, tasks);
}

std::string
ScalewayServerHelper::extractJobId(ServerMessage &postResponse) {
  if (postResponse.contains("id"))
    return postResponse["id"].get<std::string>();
  if (postResponse.contains("job_id"))
    return postResponse["job_id"].get<std::string>();
  throw std::runtime_error("Job submission failed");
}

std::string
ScalewayServerHelper::constructGetJobPath(std::string &jobId) {
  return m_qaasClient->getJobUrl(jobId);
}

std::string ScalewayServerHelper::constructGetJobPath(
    ServerMessage &postResponse) {
  std::string jobId = extractJobId(postResponse);
  return m_qaasClient->getJobUrl(jobId);
}

std::chrono::microseconds ScalewayServerHelper::nextResultPollingInterval(
    ServerMessage &postResponse) {
  return std::chrono::microseconds(1000000);
}

bool ScalewayServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::string status = getJobResponse.value("status", "unknown");

  if (status == "error") {
    std::string err = getJobResponse.contains("result")
                          ? getJobResponse["result"].value("error_message", "Unknown error")
                          : "Unknown";
    throw std::runtime_error("Scaleway Job Error: " + err);
  }

  return (status == "completed" || status == "cancelled" ||
          status == "cancelling");
}

cudaq::sample_result
ScalewayServerHelper::processResults(ServerMessage &postJobResponse,
                                     std::string &jobId) {
  auto jobResults = m_qaasClient->listJobResults(jobId);

  if (jobResults.empty()) {
    throw std::runtime_error("Job done but empty results.");
  }

  auto firstResult = jobResults[0];

  std::string rawPayload;

  if (firstResult.has_inline_result()) {
    rawPayload = firstResult.result;
  } else if (firstResult.has_download_url()) {
    RestClient client;
    rawPayload = client.getRawText(firstResult.url, "", backendConfig, true);
  } else {
    throw std::runtime_error(
        "invalid: empty 'result' and 'url' fields to get result.");
  }

  try {
    qio::QuantumProgramResult qioResult =
        qio::QuantumProgramResult::fromJson(rawPayload);

    auto sampleResult = qioResult.toCudaqSampleResult();

    return sampleResult;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Error while parsing result: " + std::string(e.what()) +
        " | payload: " + rawPayload);
  }
}

std::string ScalewayServerHelper::ensureSessionIsActive() {
  if (!m_sessionId.empty()) {
    try {
      qaas::v1alpha1::Session session = m_qaasClient->getSession(m_sessionId);
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
    auto platforms = m_qaasClient->listPlatforms(m_targetPlatformName);

    if (platforms.empty()) {
      throw std::runtime_error("No platforms found with name: " +
                               m_targetPlatformName);
    }

    auto platform = platforms[0];

    CUDAQ_INFO("Creating session on Scaleway platform {} (id={})",
               platform.name, platform.id);

    auto session = m_qaasClient->createSession(
        platform.id, m_sessionName,
        m_sessionDeduplicationId,
        "", // No model id
        m_sessionMaxDuration,
        m_sessionMaxIdleDuration,
        ""); // No parameters

    if (session.id.empty()) {
      throw std::runtime_error("Failed to create Scaleway session");
    }

    m_sessionId = session.id;
  }

  return m_sessionId;
}

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ScalewayServerHelper, scaleway)
