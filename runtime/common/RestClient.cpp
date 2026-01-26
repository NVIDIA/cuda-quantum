/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RestClient.h"
#include "FmtCore.h"
#include "Logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cpr/cpr.h>

namespace cudaq {
constexpr long validHttpCode = 205;

RestClient::RestClient() : sslOptions(std::make_unique<cpr::SslOptions>()) {
  auto caInfo = [&]() -> std::string {
    if (auto *curlCABundleStr = getenv("CURL_CA_BUNDLE")) {
      if (std::filesystem::exists(curlCABundleStr))
        return curlCABundleStr;
      else
        CUDAQ_INFO(
            "{} does not exist. Will fall back on CUDA-Q installed certs",
            curlCABundleStr);
    }
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto certPath = cudaqLibPath.parent_path().parent_path() / "cacert.pem";
    if (std::filesystem::exists(certPath))
      return certPath.string();
    CUDAQ_INFO(
        "{} does not exist, so we will rely on CURL finding the correct "
        "certificate authority bundles. If this does not work, try setting "
        "the CURL_CA_BUNDLE environment variable to a valid path to a CA "
        "Bundle file, like one downloaded from here: "
        "https://curl.se/ca/cacert.pem.",
        certPath.string());
    return "";
  }();

  if (!caInfo.empty())
    sslOptions->SetOption(cpr::ssl::CaInfo(std::move(caInfo)));
}

// Must define this in the cpp file instead of the header file
// because CPR headers aren't included in RestClient.h.
RestClient::~RestClient() = default;

nlohmann::json
RestClient::post(const std::string_view remoteUrl, const std::string_view path,
                 nlohmann::json &post,
                 std::map<std::string, std::string> &headers,
                 bool enableLogging, bool enableSsl,
                 const std::map<std::string, std::string> &cookies,
                 std::map<std::string, std::string> *cookiesOut) {
  if (headers.empty())
    headers.insert(std::make_pair("Content-type", "application/json"));

  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});

  cpr::Cookies cprCookies;
  for (const auto &kv : cookies)
    cprCookies.emplace_back({kv.first, kv.second});

  // Allow caller to disable logging for things like passwords/tokens
  if (enableLogging)
    CUDAQ_INFO("Posting to {}/{} with data = {}", remoteUrl, path, post.dump());

  auto actualPath = std::string(remoteUrl) + std::string(path);
  auto r = cpr::Post(cpr::Url{actualPath}, cpr::Body(post.dump()), cprHeaders,
                     cpr::VerifySsl(enableSsl), *sslOptions, cprCookies);

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP POST Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);

  // Update the cookies map
  if (cookiesOut)
    for (const auto &cookie : r.cookies)
      (*cookiesOut)[cookie.GetName()] = cookie.GetValue();

  return nlohmann::json::parse(r.text);
}

void RestClient::put(const std::string_view remoteUrl,
                     const std::string_view path, nlohmann::json &putData,
                     std::map<std::string, std::string> &headers,
                     bool enableLogging, bool enableSsl,
                     const std::map<std::string, std::string> &cookies) {
  if (headers.empty())
    headers.insert(std::make_pair("Content-type", "application/json"));

  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});
  cpr::Cookies cprCookies;
  for (const auto &kv : cookies)
    cprCookies.emplace_back({kv.first, kv.second});
  // Allow caller to disable logging for things like passwords/tokens
  if (enableLogging)
    CUDAQ_INFO("Putting to {}/{} with data = {}", remoteUrl, path,
               putData.dump());

  auto actualPath = std::string(remoteUrl) + std::string(path);
  auto r = cpr::Put(cpr::Url{actualPath}, cpr::Body(putData.dump()), cprHeaders,
                    cpr::VerifySsl(enableSsl), *sslOptions, cprCookies);

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP PUT Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);
}

std::string RestClient::getRawText(
    const std::string_view remoteUrl, const std::string_view path,
    std::map<std::string, std::string> &headers, bool enableSsl,
    const std::map<std::string, std::string> &cookies) {
  if (headers.empty())
    headers.insert(std::make_pair("Content-type", "application/json"));

  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});
  cpr::Cookies cprCookies;
  for (const auto &kv : cookies)
    cprCookies.emplace_back({kv.first, kv.second});
  cpr::Parameters cprParams;
  auto actualPath = std::string(remoteUrl) + std::string(path);
  auto r = cpr::Get(cpr::Url{actualPath}, cprHeaders, cprParams,
                    cpr::VerifySsl(enableSsl), *sslOptions, cprCookies);

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP GET Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);
  return r.text;
}

nlohmann::json
RestClient::get(const std::string_view remoteUrl, const std::string_view path,
                std::map<std::string, std::string> &headers, bool enableSsl,
                const std::map<std::string, std::string> &cookies) {
  return nlohmann::json::parse(
      getRawText(remoteUrl, path, headers, enableSsl, cookies));
}

void RestClient::del(const std::string_view remoteUrl,
                     const std::string_view path,
                     std::map<std::string, std::string> &headers,
                     bool enableLogging, bool enableSsl,
                     const std::map<std::string, std::string> &cookies) {
  cpr::Header cprHeaders;
  for (auto &kv : headers)
    cprHeaders.insert({kv.first, kv.second});
  cpr::Cookies cprCookies;
  for (const auto &kv : cookies)
    cprCookies.emplace_back({kv.first, kv.second});
  cpr::Parameters cprParams;
  auto actualPath = std::string(remoteUrl) + std::string(path);
  if (enableLogging)
    CUDAQ_INFO("Delete resource at path {}/{}", remoteUrl, path);
  auto r = cpr::Delete(cpr::Url{actualPath}, cprHeaders, cprParams,
                       cpr::VerifySsl(enableSsl), *sslOptions, cprCookies);

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP DELETE Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);
}

void RestClient::download(const std::string_view remoteUrl,
                          const std::string &filePath, bool enableLogging,
                          bool enableSsl,
                          const std::map<std::string, std::string> &cookies) {
  cpr::Cookies cprCookies;
  for (const auto &kv : cookies)
    cprCookies.emplace_back({kv.first, kv.second});
  auto r = cpr::Get(cpr::Url{std::string(remoteUrl)}, cpr::Header{},
                    cpr::Parameters{}, cpr::VerifySsl(enableSsl), *sslOptions,
                    cprCookies);

  if (r.status_code > validHttpCode || r.status_code == 0)
    throw std::runtime_error("HTTP Download Error - status code " +
                             std::to_string(r.status_code) + ": " +
                             r.error.message + ": " + r.text);

  if (enableLogging)
    CUDAQ_INFO("Downloading {} bytes from {} to file {}", r.text.size(),
               remoteUrl, filePath);

  try {
    // Write the downloaded content to file.
    std::ofstream outfile(filePath, std::ofstream::binary | std::ios::out);
    outfile.write(r.text.c_str(), r.text.size());
    outfile.close();
  } catch (std::exception &e) {
    // Rethrow it with a descriptive message
    throw std::runtime_error(fmt::format(
        "Failed to write downloaded contents to file {}. Exception: {}.",
        filePath, e.what()));
  }
}
} // namespace cudaq
