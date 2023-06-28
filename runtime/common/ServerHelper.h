/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json.hpp"

#include "ExecutionContext.h"
#include "Future.h"
#include "MeasureCounts.h"
#include "Registry.h"

namespace cudaq {

/// @brief Typedef for a mapping of key-values describing the remote server
/// configuration.
using BackendConfig = std::map<std::string, std::string>;

/// @brief Every kernel execution has a name and a compiled code representation
struct KernelExecution {
  std::string name;
  std::string code;
  KernelExecution(std::string &n, std::string &c) : name(n), code(c) {}
};

/// @brief Responses / Submissions to the Server are modeled via JSON
using ServerMessage = nlohmann::json;

/// @brief Each REST interaction will require headers
using RestHeaders = std::map<std::string, std::string>;

// A Server Job Payload consists of a job post URL path, the headers,
// and a vector of related Job JSON messages.
using ServerJobPayload =
    std::tuple<std::string, RestHeaders, std::vector<ServerMessage>>;

/// @brief The ServerHelper is a Plugin type that abstracts away the
/// server-specific information needed for submitting quantum jobs
/// to a remote server. It enables clients to create server-specific job
/// payloads, extract job ids, and query / request job results. Moreover it
/// provides a hook for extracting results from a server response.
class ServerHelper : public registry::RegisteredType<ServerHelper> {
protected:
  /// @brief All ServerHelpers can be configured at the nvq++ command line.
  /// This map holds those configuration key-values
  BackendConfig backendConfig;

  /// @brief The number of shots to execute
  std::size_t shots = 100;

public:
  ServerHelper() = default;
  virtual ~ServerHelper() = default;
  virtual const std::string name() const = 0;

  /// @brief Set the number of shots to execute
  void setShots(std::size_t s) { shots = s; }

  /// @brief Set the server configuration.
  virtual void initialize(BackendConfig config) = 0;

  /// @brief Return the current server configuration
  /// @return
  BackendConfig getConfig() { return backendConfig; }

  /// @brief Return the POST/GET required headers.
  /// @return
  virtual RestHeaders getHeaders() = 0;

  /// @brief Given a vector of compiled quantum codes for submission
  /// create and return the Job payload that is compatible with this server.
  virtual ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) = 0;

  /// @brief Extract the job id from the server response from posting the job.
  virtual std::string extractJobId(ServerMessage &postResponse) = 0;

  /// @brief Get the specific path required to retrieve job results.
  /// Construct specifically from the job id.
  virtual std::string constructGetJobPath(std::string &jobId) = 0;

  /// @brief Get the specific path required to retrieve job results. Construct
  /// from the full server response message.
  virtual std::string constructGetJobPath(ServerMessage &postResponse) = 0;

  /// @brief Return true if the job is done.
  virtual bool jobIsDone(ServerMessage &getJobResponse) = 0;

  /// @brief Given a successful job and the success response,
  /// retrieve the results and map them to a sample_result.
  /// @param postJobResponse
  /// @return
  virtual cudaq::sample_result
  processResults(ServerMessage &postJobResponse) = 0;
};
} // namespace cudaq
