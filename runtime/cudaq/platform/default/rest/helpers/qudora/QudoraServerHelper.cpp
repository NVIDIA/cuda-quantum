/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <fstream>
#include <iostream>
#include <thread>
#include <zlib.h>

namespace cudaq {

/// Search for the API key
std::string searchAPIKeyQudora();

/// @brief The QudoraServerHelper implements the ServerHelper interface
/// to map Job requests and Job result retrievals actions from the calling
/// Executor to the specific schema required by the remote Qudora REST
/// server.
class QudoraServerHelper : public ServerHelper, public QirServerHelper {
protected:
  /// @brief The base URL
  std::string baseUrl = "https://api.qudora.com/jobs/";
  /// @brief The machine we are targeting
  std::string machine = "Qamelion";
  /// @brief Backend settings to be sent to the remote backend.
  nlohmann::json backend_settings = nullptr;

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "qudora"; }
  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    // Set the machine
    auto iter = backendConfig.find("machine");
    if (iter != backendConfig.end())
      machine = iter->second;

    // Set the backend_settings
    iter = backendConfig.find("backend_settings");
    if (iter != backendConfig.end()) {
      try {
        backend_settings = nlohmann::json::parse(iter->second);
      } catch (nlohmann::json::parse_error &e) {
        throw std::runtime_error("Failed to parse backend_settings: " +
                                 std::string(e.what()));
      }
    }

    // Set an alternate base URL if provided
    iter = backendConfig.find("url");
    if (iter != backendConfig.end()) {
      baseUrl = iter->second;
      if (!baseUrl.ends_with("/"))
        baseUrl += "/";
    }

    parseConfigForCommonParams(config);
  }

  /// @brief Create a job payload for the provided quantum codes
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Get the jobs results polling interval.
  /// @return
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) override;

  /// @brief Extract QIR output data from the server's response to a job
  std::string extractOutputLog(ServerMessage &postJobResponse,
                               std::string &jobId) override;
};

std::string zlibDecompress(const std::vector<char> &compressed) {
  z_stream zs{};
  zs.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(compressed.data()));
  zs.avail_in = static_cast<uInt>(compressed.size());

  if (inflateInit(&zs) != Z_OK)
    throw std::runtime_error("inflateInit failed");

  int ret;
  char outbuffer[32768];
  std::string output;

  do {
    zs.next_out = reinterpret_cast<Bytef *>(outbuffer);
    zs.avail_out = sizeof(outbuffer);

    ret = inflate(&zs, 0);

    if (output.size() < zs.total_out) {
      output.append(outbuffer, zs.total_out - output.size());
    }

  } while (ret == Z_OK);

  inflateEnd(&zs);

  if (ret != Z_STREAM_END)
    throw std::runtime_error("zlib decompression failed");

  return output;
}

ServerJobPayload
QudoraServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Construct the job itself
  ServerMessage j;
  j["language"] = "QIR_BITCODE";
  j["target"] = machine;

  std::vector<ServerMessage> messages;
  for (auto &circuitCode : circuitCodes) {
    j["name"] = "CUDA-Q " + circuitCode.name;
    j["shots"] = {shots};
    j["input_data"] = {circuitCode.code};
    j["backend_settings"] = backend_settings;
    messages.push_back(j);
  }

  // Get the headers
  RestHeaders headers = getHeaders();

  cudaq::info(
      "Created job payload for Qudora, language is QIR 1.0, targeting {}",
      machine);

  // return the payload
  return std::make_tuple(baseUrl, headers, messages);
}

std::string QudoraServerHelper::extractJobId(ServerMessage &postResponse) {
  std::string id = to_string(postResponse);
  return id;
}

std::string
QudoraServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  std::string job_id = extractJobId(postResponse);
  return constructGetJobPath(job_id);
}

std::chrono::microseconds
QudoraServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  return std::chrono::milliseconds(200);
}

std::string QudoraServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + "?job_id=" + jobId + "&include_results=True";
}

bool QudoraServerHelper::jobIsDone(ServerMessage &getJobResponse) {

  auto status = getJobResponse[0]["status"].get<std::string>();

  if (status == "Failed") {
    throw std::runtime_error(
        "Job failed to execute. See QUDORA Cloud for more details.");
  } else if (status == "Canceled" || status == "Deleted" ||
             status == "Cancelling") {
    throw std::runtime_error("Job was cancelled.");
  }

  return status == "Completed";
}

cudaq::sample_result
QudoraServerHelper::processResults(ServerMessage &postJobResponse,
                                   std::string &jobId) {

  auto qirResults = extractOutputLog(postJobResponse, jobId);
  return createSampleResultFromQirOutput(qirResults);
}

std::string QudoraServerHelper::extractOutputLog(ServerMessage &postJobResponse,
                                                 std::string &jobId) {
  CUDAQ_DBG("postJobResponse: {}", postJobResponse.dump());
  CUDAQ_INFO("jobId: {}", jobId);
  auto compressedQirResultsB64 =
      postJobResponse[0]["qir_result"][0].get<std::string>();
  std::vector<char> compressQirResults;
  if (llvm::decodeBase64(compressedQirResultsB64, compressQirResults))
    throw std::runtime_error(
        "Invalid results received. See QUDORA Cloud for more details.");
  auto qirResults = zlibDecompress(compressQirResults);
  return qirResults;
}

RestHeaders QudoraServerHelper::getHeaders() {
  std::string apiKey = searchAPIKeyQudora();
  std::map<std::string, std::string> headers{
      {"Authorization", "Bearer " + apiKey},
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"Accept", "*/*"}};
  return headers;
}

/// Search for the API key
std::string searchAPIKeyQudora() {
  // Allow someone to tweak this with an environment variable
  if (auto creds = std::getenv("CUDAQ_QUDORA_CREDENTIALS")) {
    return std::string(creds);
  } else {
    throw std::runtime_error("Cannot find QUDORA API key in "
                             "CUDAQ_QUDORA_CREDENTIALS environment variable.");
  }
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QudoraServerHelper, qudora)
