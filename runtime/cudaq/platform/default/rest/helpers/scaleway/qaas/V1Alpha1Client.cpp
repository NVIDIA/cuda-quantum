/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2026 Scaleway                                      *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "V1Alpha1Client.h"
#include "cudaq/runtime/logger/logger.h"
#include <stdexcept>
#include <regex>
#include <sstream>
#include <iostream>

using json = nlohmann::json;
using namespace cudaq::qaas::v1alpha1;

long parseDurationToSeconds(const std::string& input) {
    std::regex re(R"((\d+)([smhd]))");
    std::smatch match;

    long total_seconds = 0;
    std::string s = input;

    while (std::regex_search(s, match, re)) {
        long value = std::stol(match[1]);
        char unit = match[2].str()[0];

        if (unit == 's') total_seconds += value;
        else if (unit == 'm') total_seconds += value * 60;
        else if (unit == 'h') total_seconds += value * 3600;
        else if (unit == 'd') total_seconds += value * 86400;

        s = match.suffix().str();
    }

    return total_seconds;
}

V1Alpha1Client::V1Alpha1Client(const std::string projectId,
                                const std::string secretKey, std::string url,
                                bool secure, bool logging) :
                                m_projectId(projectId),
                                m_secretKey(secretKey),
                                m_secure(secure),
                                m_logging(logging) {
    auto built_url = url.empty() ? DEFAULT_URL : url;
    m_baseUrl = built_url + "/" + DEFAULT_BASE_PATH;

    CUDAQ_INFO("Initializing Scaleway QaaS Client with URL: {}", m_baseUrl);
}

std::map<std::string, std::string> V1Alpha1Client::getHeaders() {
  return {{"X-Auth-Token", m_secretKey}, {"Content-Type", "application/json"}};
}

std::string V1Alpha1Client::getJobsUrl() { return m_baseUrl + "/jobs"; }

std::string V1Alpha1Client::getJobUrl(const std::string &jobId) {
  return m_baseUrl + "/jobs/" + jobId;
}

std::string V1Alpha1Client::getJobResultsUrl(const std::string &jobId) {
  return m_baseUrl + "/jobs/" + jobId + "/results";
}

Platform V1Alpha1Client::getPlatform(const std::string &platformId) {
  auto headers = getHeaders();
  std::string path = "/platforms/" + platformId;

  try {
    auto response = m_client.get(m_baseUrl, path, headers, m_secure);
    return response.get<Platform>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during get platform " + platformId + ": " +
                             e.what());
  }
}

std::vector<Platform>
V1Alpha1Client::listPlatforms(const std::string platformName) {
  auto headers = getHeaders();
  std::string path = "/platforms";

  if (!platformName.empty()) {
    path += "?name=" + platformName;
  }

  try {
    auto response = m_client.get(m_baseUrl, path, headers, m_secure);

    CUDAQ_INFO("List Platform response: {}", response.dump());

    if (response.contains("platforms")) {
      return response["platforms"].get<std::vector<Platform>>();
    }
    return {};
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during list platforms: " +
                             std::string(e.what()));
  }
}

Session V1Alpha1Client::createSession(const std::string &platformId,
                                      std::string name,
                                      std::string deduplicationId,
                                      const std::string &modelId,
                                      std::string maxDuration,
                                      std::string maxIdleDuration,
                                      std::string parameters) {
  auto headers = getHeaders();
  json payload;

  payload["project_id"] = m_projectId;
  payload["platform_id"] = platformId;

  if (!name.empty())
    payload["name"] = name;
  if (!deduplicationId.empty())
    payload["deduplication_id"] = deduplicationId;
  if (!modelId.empty())
    payload["model_id"] = modelId;

  payload["max_duration"] = std::to_string(parseDurationToSeconds(maxDuration)) + "s";
  payload["max_idle_duration"] = std::to_string(parseDurationToSeconds(maxIdleDuration)) + "s";

  if (!parameters.empty()) {
    try {
      payload["parameters"] = json::parse(parameters);
    } catch (...) {
      payload["parameters"] = parameters;
    }
  }

  try {
    CUDAQ_INFO("Create Session request: {}", payload.dump());

    auto response = m_client.post(m_baseUrl, "/sessions", payload, headers,
                                  m_logging, m_secure);

    CUDAQ_INFO("Create Session response: {}", response.dump());

    return response.get<Session>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during session creation: " +
                             std::string(e.what()));
  }
}

Session V1Alpha1Client::getSession(const std::string &sessionId) {
  auto headers = getHeaders();
  std::string path = "/sessions/" + sessionId;

  try {
    auto response = m_client.get(m_baseUrl, path, headers, m_secure);

    CUDAQ_INFO("Get Session response: {}", response.dump());

    return response.get<Session>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during get session " + sessionId + ": " +
                             e.what());
  }
}

Job V1Alpha1Client::createJob(const std::string &sessionId,
                              const std::string &modelId, std::string name) {
  auto headers = getHeaders();
  json payload;

  payload["session_id"] = sessionId;
  payload["model_id"] = modelId;
  payload["name"] = name;

  try {
    CUDAQ_INFO("Create Job request: {}", payload.dump());

    auto response = m_client.post(m_baseUrl, "/jobs", payload, headers,
                                  m_logging, m_secure);

    CUDAQ_INFO("Create Job response: {}", response.dump());

    return response.get<Job>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during job submission: " +
                             std::string(e.what()));
  }
}

Job V1Alpha1Client::getJob(const std::string &jobId) {
  auto headers = getHeaders();
  std::string path = "/jobs/" + jobId;

  try {
    auto response = m_client.get(m_baseUrl, path, headers, m_secure);
    return response.get<Job>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during get job " + jobId + ": " + e.what());
  }
}

std::vector<JobResult>
V1Alpha1Client::listJobResults(const std::string &jobId) {
  auto headers = getHeaders();
  std::string path = "/jobs/" + jobId + "/results";

  try {
    auto response = m_client.get(m_baseUrl, path, headers, m_secure);

    if (response.contains("job_results")) {
      return response["job_results"].get<std::vector<JobResult>>();
    }
    return {};
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during list job results " + jobId + ": " +
                             e.what());
  }
}

Model V1Alpha1Client::createModel(const std::string &data) {
  auto headers = getHeaders();
  json payload;

  payload["project_id"] = m_projectId;
  payload["payload"] = data;

  try {
    CUDAQ_INFO("Create Model request: {}", payload.dump());

    auto response = m_client.post(m_baseUrl, "/models", payload, headers,
                                  m_logging, m_secure);

    CUDAQ_INFO("Create Model response: {}", response.dump());

    return response.get<Model>();
  } catch (const std::exception &e) {
    throw std::runtime_error("fail during model creation: " +
                             std::string(e.what()));
  }
}
