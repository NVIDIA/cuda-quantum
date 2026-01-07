/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "ExecutionContext.h"
#include "Future.h"
#include "Registry.h"
#include "RuntimeTarget.h"
#include "SampleResult.h"
#include "common/RecordLogParser.h"
#include "nlohmann/json.hpp"
#include <filesystem>

namespace cudaq {

/// @brief Typedef for a mapping of key-values describing the remote server
/// configuration.
using BackendConfig = std::map<std::string, std::string>;

/// @brief Every kernel execution has a name, compiled code representation, and
/// (optionally) an output_names mapping showing how each Result maps back
/// to the original program's Qubits.
struct KernelExecution {
  std::string name;
  std::string code;
  nlohmann::json output_names;
  std::vector<std::size_t> mapping_reorder_idx;
  nlohmann::json user_data;
  KernelExecution(std::string &n, std::string &c, nlohmann::json &o,
                  std::vector<std::size_t> &m)
      : name(n), code(c), output_names(o), mapping_reorder_idx(m) {}
  KernelExecution(std::string &n, std::string &c, nlohmann::json &o,
                  std::vector<std::size_t> &m, nlohmann::json &ud)
      : name(n), code(c), output_names(o), mapping_reorder_idx(m),
        user_data(ud) {}
};

/// @brief Responses / Submissions to the Server are modeled via JSON
using ServerMessage = nlohmann::json;

/// @brief Each REST interaction will require headers
using RestHeaders = std::map<std::string, std::string>;

/// @brief Cookies are also a map of key-values
using RestCookies = std::map<std::string, std::string>;

// A Server Job Payload consists of a job post URL path, the headers,
// and a vector of related Job JSON messages.
using ServerJobPayload =
    std::tuple<std::string, RestHeaders, std::vector<ServerMessage>>;

/// @brief Information about a result coming from a backend
struct ResultInfoType {
  std::size_t qubitNum;
  std::string registerName;
};

/// @brief Results information, indexed by 0-based result number
using OutputNamesType = std::map<std::size_t, ResultInfoType>;

/// @brief The ServerHelper is a Plugin type that abstracts away the
/// server-specific information needed for submitting quantum jobs
/// to a remote server. It enables clients to create server-specific job
/// payloads, extract job ids, and query / request job results. Moreover it
/// provides a hook for extracting results from a server response.
class ServerHelper : public registry::RegisteredType<ServerHelper> {
protected:
  /// @brief All ServerHelpers can be configured at the `nvq++` command line.
  /// This map holds those configuration key-values
  BackendConfig backendConfig;

  /// @brief The number of shots to execute
  std::size_t shots = 100;

  /// @brief Parse a `config` for common parameters in a server helper (i.e.
  /// `outputNames` and `reorderIdx`)
  void parseConfigForCommonParams(const BackendConfig &config);

  /// @brief Output names indexed by jobID/taskID
  std::map<std::string, OutputNamesType> outputNames;

  /// @brief Reordering indices indexed by jobID/taskID (used by mapping pass)
  std::map<std::string, std::vector<std::size_t>> reorderIdx;

  /// @brief  Information about the runtime target managing this server helper.
  RuntimeTarget runtimeTarget;

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

  /// @brief Return the cookies required for the request.
  // By default, no cookies.
  virtual RestCookies getCookies() { return {}; }

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

  /// @brief Get the jobs results polling interval.
  /// @return
  virtual std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) {
    return std::chrono::microseconds(100);
  }

  /// @brief Return true if the job is done.
  virtual bool jobIsDone(ServerMessage &getJobResponse) = 0;

  /// @brief Given a successful job and the success response,
  /// retrieve the results and map them to a sample_result.
  /// @param postJobResponse
  /// @param jobId
  /// @return
  virtual cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                              std::string &jobId) = 0;

  /// @brief Adjust the compiler pass pipeline (if desired)
  virtual void updatePassPipeline(const std::filesystem::path &platformPath,
                                  std::string &passPipeline) {}

  /// @brief Set the runtime target information
  void setRuntimeTarget(const RuntimeTarget &target) { runtimeTarget = target; }
};

/// @brief Server helper interface for QIR-based output servers.
class QirServerHelper {
public:
  QirServerHelper() = default;
  virtual ~QirServerHelper() = default;
  /// @brief Given a successful job and the success response,
  /// retrieve the QIR output log
  /// @param postJobResponse
  /// @param jobId
  /// @return QIR output log
  virtual std::string extractOutputLog(ServerMessage &postJobResponse,
                                       std::string &jobId) = 0;

  /// @brief Create a sampling result from the QIR output log
  cudaq::sample_result
  createSampleResultFromQirOutput(const std::string &qirOutputLog) {
    // Parse the QIR output log
    cudaq::RecordLogParser parser;
    parser.parse(qirOutputLog);

    // Get the buffer and length of buffer (in bytes) from the parser.
    auto *origBuffer = parser.getBufferPtr();
    std::size_t bufferSize = parser.getBufferSize();
    char *buffer = static_cast<char *>(malloc(bufferSize));
    std::memcpy(buffer, origBuffer, bufferSize);

    std::vector<std::vector<bool>> results = {
        reinterpret_cast<std::vector<bool> *>(buffer),
        reinterpret_cast<std::vector<bool> *>(buffer + bufferSize)};
    const auto numShots = results.size();
    // Create the counts dictionary
    cudaq::CountsDictionary globalCounts;
    std::vector<std::string> globalSequentialData;
    globalSequentialData.reserve(numShots);
    for (const auto &shotResult : results) {
      // Each shot is an array of tagged results
      std::string bitString;
      for (const auto &bitVal : shotResult) {
        bitString.append(bitVal ? "1" : "0");
      }
      // Global register results
      globalCounts[bitString]++;
      globalSequentialData.push_back(bitString);
    }

    // Add the global register results
    cudaq::ExecutionResult result{globalCounts, GlobalRegisterName};
    result.sequentialData = globalSequentialData;
    return cudaq::sample_result({result});
  }
};
} // namespace cudaq
