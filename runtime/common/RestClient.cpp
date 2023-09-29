/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RestClient.h"
#include "Logger.h"
#include <cpr/cpr.h>

namespace cudaq {
constexpr long validHttpCode = 205;

nlohmann::json RestClient::post(const std::string_view remoteUrl,
                                const std::string_view path,
                                nlohmann::json &post,
                                std::map<std::string, std::string> &headers,
                                bool enableLogging) {
  if (headers.empty())
    headers.insert(std::make_pair("Content-type", "application/json"));

  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});

  // Allow caller to disable logging for things like passwords/tokens
  if (enableLogging)
    cudaq::info("Posting to {}/{} with data = {}", remoteUrl, path,
                post.dump());

  auto actualPath = std::string(remoteUrl) + std::string(path);
  auto r = cpr::Post(cpr::Url{actualPath}, cpr::Body(post.dump()), cprHeaders,
                     cpr::VerifySsl(false));

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP POST Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);

  return nlohmann::json::parse(r.text);
}

nlohmann::json RestClient::get(const std::string_view remoteUrl,
                               const std::string_view path,
                               std::map<std::string, std::string> &headers) {
  if (headers.empty())
    headers.insert(std::make_pair("Content-type", "application/json"));

  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});

  cpr::Parameters cprParams;
  auto actualPath = std::string(remoteUrl) + std::string(path);
  auto r = cpr::Get(cpr::Url{actualPath}, cprHeaders, cprParams,
                    cpr::VerifySsl(false));

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP GET Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);

  return nlohmann::json::parse(r.text);
}

} // namespace cudaq
