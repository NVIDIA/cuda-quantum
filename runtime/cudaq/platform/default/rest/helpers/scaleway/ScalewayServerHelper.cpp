/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                                     *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "ScalewayServerHelper.h"
#include "cudaq/runtime/logger/logger.h"
#include "common/RestClient.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <map>
#include <sstream>

using json = nlohmann::json;
using namespace cudaq;

std::string getEnv(const std::string &envKey) {
  if (envKey.empty())
    return "";

  auto envVar = std::getenv(envKey.c_str());

  // Handles nullptr
  std::string var(envVar ? envVar : "");

  return var;
}

std::string getValueOrDefault(const BackendConfig &config,
                              const std::string &key,
                              const std::string &envKey,
                              const std::string &defaultValue) {
  auto it = config.find(key);

  // If no provided value, look from env variables
  auto envValue = getEnv(envKey);

  auto providedValue = (it != config.end()) ? it->second : envValue;

  // If still no value, we apply the default SDK value
  return !providedValue.empty() ? providedValue : defaultValue;
}

std::string serializeParametersToQio(size_t nb_shots, std::string output_names) {
  // auto output_names = circuitCode.output_names.dump();
  // backendConfig["output_names." + model.id] = output_names;

  CUDAQ_INFO("Output names {}", output_names);

  // OutputNamesType jobOutputNames;
  // nlohmann::json outputNamesJson = nlohmann::json::parse(output_names);

  // CUDAQ_INFO("Create output names {} {}", taskId, output_names);

  // for (const auto &el : outputNamesJson[0]) {
  //   auto result = el[0].get<std::size_t>();
  //   auto qubitNum = el[1][0].get<std::size_t>();
  //   auto registerName = el[1][1].get<std::string>();

  //   CUDAQ_INFO("Create register res:{}, nb:{}, name:{}", result, qubitNum, registerName);

  //   jobOutputNames[result] = {qubitNum, registerName};
  // }

  // outputNames[taskId] = jobOutputNames;

  // setOutputNames(model.id, output_names);

  json options;
  options["output_names"] = output_names;
  qio::QuantumComputationParameters parameters(nb_shots, options);

  return parameters.toJson().dump();
}

std::string serializeKernelToQio(const std::string &code) {
  qio::QuantumProgram program(code,qio::QuantumProgramSerializationFormat::QASM_V2,
                              qio::CompressionFormat::ZLIB_BASE64_V1);

  std::vector<qio::QuantumProgram> programs = {program};

  qio::QuantumComputationModel model(programs);

  return model.toJson().dump();
}

void ScalewayServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing Scaleway Server Helper");

  backendConfig = config;

  m_qaasClient = std::make_unique<qaas::v1alpha1::V1Alpha1Client>(
                      getValueOrDefault(config, "project_id", "SCW_PROJECT_ID", ""),
                      getValueOrDefault(config, "secret_key", "SCW_SECRET_KEY", ""),
                      getValueOrDefault(config, "url", "SCW_API_URL", ""));

  m_targetPlatformName = getValueOrDefault(config, "machine", "", DEFAULT_PLATFORM_NAME);
  m_sessionMaxDuration = getValueOrDefault(config, "max_duration", "", DEFAULT_MAX_DURATION);
  m_sessionMaxIdleDuration = getValueOrDefault(config, "max_idle_duration", "", DEFAULT_MAX_IDLE_DURATION);
  m_sessionDeduplicationId = getValueOrDefault(config, "deduplication_id", "", "");
  m_sessionName = getValueOrDefault(config, "name", "", "qs-cudaq-" + std::to_string(std::rand()));

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

  CUDAQ_INFO("Creating job for Scaleway, "
             "targeting platform {}",
             m_targetPlatformName);

  for (auto &circuitCode : circuitCodes) {
    ServerMessage taskRequest;
    CUDAQ_INFO("Job name {}", circuitCode.name);

    std::string qioPayload = serializeKernelToQio(circuitCode.code);
    CUDAQ_INFO("Attached payload {}", qioPayload);

    std::string qioParams = serializeParametersToQio(shots, circuitCode.output_names.dump());
    CUDAQ_INFO("Attached parameters {}", qioParams);

    auto model = m_qaasClient->createModel(qioPayload);
    CUDAQ_INFO("Created model {}", model.id);

    taskRequest["model_id"] = model.id;
    taskRequest["session_id"] = m_sessionId;
    taskRequest["name"] = circuitCode.name;
    taskRequest["parameters"] = qioParams;

    tasks.push_back(taskRequest);
  }

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
  return std::chrono::microseconds(100000);
}

bool ScalewayServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::string status = getJobResponse.value("status", "Unknown status");

  if (status == "error") {
    std::string err = getJobResponse.value("progress_message", "Unknown error");
    throw std::runtime_error("Scaleway Job Error: " + err);
  }

  return (status == "completed" || status == "cancelled" ||
          status == "cancelling");
}

cudaq::sample_result
ScalewayServerHelper::processResults(ServerMessage &postJobResponse,
                                     std::string &jobId) {
  CUDAQ_INFO("Post-processing results for job {}", jobId);

  auto jobResults = m_qaasClient->listJobResults(jobId);

  if (jobResults.empty()) {
    throw std::runtime_error("Job done but no result.");
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

  CUDAQ_INFO("Get raw results for job {}: {}", jobId, rawPayload);

  try {
    auto jsonPayload = json::parse(rawPayload);

    auto qioResult = qio::QuantumProgramResult::fromJson(jsonPayload);

    auto sampleResult = qioResult.toCudaqSampleResult();

    std::vector<ExecutionResult> execResults;

    auto job = m_qaasClient->getJob(jobId);

    CUDAQ_INFO("job param {}", job.parameters);

    auto jsonParameters = json::parse(job.parameters);

    auto params = qio::QuantumComputationParameters::fromJson(jsonParameters);
    auto options = params.options();

    CUDAQ_INFO("options {}", options);

    auto outputNamesStr = options["output_names"].get<std::string>();

    CUDAQ_INFO("outputNamesStr {}", outputNamesStr);

    auto outputNamesJson = json::parse(outputNamesStr);
    OutputNamesType jobOutputNames;

    for (const auto &el : outputNamesJson[0]) {
      auto result = el[0].get<std::size_t>();
      auto qubitNum = el[1][0].get<std::size_t>();
      auto registerName = el[1][1].get<std::string>();

      CUDAQ_INFO("Create register res:{}, nb:{}, name:{}", result, qubitNum, registerName);

      jobOutputNames[result] = {qubitNum, registerName};
    }

    // Get a reduced list of qubit numbers that were in the original program
    // so that we can slice the output data and extract the bits that the user
    // was interested in. Sort by QIR qubit number.
    std::vector<std::size_t> qubitNumbers;
    qubitNumbers.reserve(jobOutputNames.size());
    for (auto &[result, info] : jobOutputNames) {
      qubitNumbers.push_back(info.qubitNum);
    }

    CUDAQ_INFO("qubitNumbers s:{} q:{}", jobOutputNames.size(), qubitNumbers);

    // For each original counts entry in the full sample results, reduce it
    // down to the user component and add to userGlobal. If qubitNumbers is empty,
    // that means all qubits were measured.
    if (qubitNumbers.empty()) {
      execResults.emplace_back(ExecutionResult{sampleResult.to_map()});
    } else {
      auto subset = sampleResult.get_marginal(qubitNumbers);
      execResults.emplace_back(ExecutionResult{subset.to_map()});
    }

    // Now add to `execResults` one register at a time
    for (const auto &[result, info] : jobOutputNames) {
      CountsDictionary regCounts;
      for (const auto &[bits, count] : sampleResult)
        regCounts[std::string{bits[info.qubitNum]}] += count;
      execResults.emplace_back(regCounts, info.registerName);
    }

    // Return a sample result including the global register and all individual
    // registers.
    auto ret = cudaq::sample_result(execResults);

    return ret;
  } catch (const std::exception &e) {
    throw std::runtime_error(
        "Error while parsing result: " + std::string(e.what()));
  }
}

std::string ScalewayServerHelper::ensureSessionIsActive() {
  if (!m_sessionId.empty()) {
    CUDAQ_INFO("Alive session id: {}", m_sessionId);

    try {
      auto session = m_qaasClient->getSession(m_sessionId);
      auto status = session.status;

      if (status == "error" || status == "stopped" || status == "stopping") {
        CUDAQ_INFO("Dead session id {} with status {}", m_sessionId, status);
        m_sessionId = "";
      } else {
        return m_sessionId;
      }
    } catch (...) {
      m_sessionId = "";
    }
  }

  if (m_sessionId.empty()) {
    CUDAQ_INFO("Searching platform with name {}", m_targetPlatformName);

    auto platforms = m_qaasClient->listPlatforms(m_targetPlatformName);

    if (platforms.empty()) {
      throw std::runtime_error("No platform found with name: " +
                               m_targetPlatformName);
    }

    auto platform = platforms[0];

    if (platform.availability == "maintenance" ||
      platform.availability == "shortage") {
        throw std::runtime_error("Target platform not available: " + platform.availability);
    }

    CUDAQ_INFO("Creating session on platform {} (id={})",
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

  CUDAQ_INFO("Active session with id: {}", m_sessionId);

  return m_sessionId;
}

// void ScalewayServerHelper::setOutputNames(const std::string &taskId,
//                                         const std::string &output_names) {
//   // Parse `output_names` into jobOutputNames.
//   // Note: See `ExtendMeasurePattern` of `CombineMeasurements.cpp
//   // for an example of how this was populated.
//   OutputNamesType jobOutputNames;
//   nlohmann::json outputNamesJson = nlohmann::json::parse(output_names);

//   CUDAQ_INFO("Create output names {} {}", taskId, output_names);

//   for (const auto &el : outputNamesJson[0]) {
//     auto result = el[0].get<std::size_t>();
//     auto qubitNum = el[1][0].get<std::size_t>();
//     auto registerName = el[1][1].get<std::string>();

//     CUDAQ_INFO("Create register res:{}, nb:{}, name:{}", result, qubitNum, registerName);

//     jobOutputNames[result] = {qubitNum, registerName};
//   }

//   outputNames[taskId] = jobOutputNames;
// }

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ScalewayServerHelper, scaleway)
