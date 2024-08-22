/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Future.h"
#include "common/Registry.h"
#include "nlohmann/json.hpp"
// #include "common/ServerHelper.h"
#include "orca_qpu.h"
#include "llvm/Support/Base64.h"

namespace cudaq {
using BackendConfig = std::map<std::string, std::string>;

/// @brief Responses / Submissions to the Server are modeled via JSON
using ServerMessage = nlohmann::json;

/// @brief Each REST interaction will require headers
using RestHeaders = std::map<std::string, std::string>;

// A Server Job Payload consists of a job post URL path, the headers,
// and a vector of related Job JSON messages.
using ServerJobPayload =
    std::tuple<std::string, RestHeaders, std::vector<ServerMessage>>;

// /// @brief Information about a result coming from a backend
// struct ResultInfoType {
//   std::size_t qubitNum;
//   std::string registerName;
// };

// /// @brief Results information, indexed by 0-based result number
// using OutputNamesType = std::map<std::size_t, ResultInfoType>;

class OrcaServerHelper
// : public ServerHelper {
    : public registry::RegisteredType<OrcaServerHelper> { 
protected:
  /// @brief All ServerHelpers can be configured at the `nvq++` command line.
  /// This map holds those configuration key-values
  BackendConfig backendConfig;

  /// @brief The number of shots to execute
  std::size_t shots = 100;

  // /// @brief Output names indexed by jobID/taskID
  // std::map<std::string, OutputNamesType> outputNames;

  /// @brief The base URL
  std::string baseUrl = "http://localhost:8080/";

  /// @brief The machine we are targeting
  std::string machine = "PT-1";

  /// @brief Time string, when the last tokens were retrieved
  std::string timeStr = "";

  /// @brief The refresh token
  std::string refreshKey = "";

  /// @brief Orca requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

public:
  OrcaServerHelper() = default;
  virtual ~OrcaServerHelper() = default;

  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const { return "orca"; }

  /// @brief Return the POST/GET required headers.
  /// @return
  RestHeaders getHeaders();

  /// @brief Return the current server configuration
  /// @return
  BackendConfig getConfig() { return backendConfig; }

  /// @brief Set the number of shots to execute
  void setShots(std::size_t s) { shots = s; }

  /// @brief Set the server configuration.
  void initialize(BackendConfig config);

  /// @brief Create a job payload for the provided TBI parameters
  ServerJobPayload createJob(cudaq::orca::TBIParameters params);

  // /// @brief Create a job payload for the provided quantum codes
  // ServerJobPayload createJob(std::vector<KernelExecution> &circuitCodes);

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse);

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse);
  std::string constructGetJobPath(std::string &jobId);

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse);

  /// @brief Given a completed job response, map back to the sample_result
  sample_result processResults(ServerMessage &postJobResponse);

    /// @brief Given a completed job response, map back to the sample_result
  sample_result processResults(ServerMessage &postJobResponse,
                                 std::string &jobID );
};

} // namespace cudaq
