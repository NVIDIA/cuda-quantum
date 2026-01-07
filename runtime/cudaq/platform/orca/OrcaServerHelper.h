/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Registry.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include "orca_qpu.h"

#include "nlohmann/json.hpp"

namespace cudaq {

class OrcaServerHelper : public ServerHelper {

protected:
  /// @brief The base URL
  std::string baseUrl = "http://localhost:8080/";

  /// @brief The machine we are targeting
  std::string machine = "PT-1";

  /// @brief Time string, when the last tokens were retrieved
  std::string timeStr = "";

  /// @brief The refresh token
  std::string refreshKey = "";

  /// @brief ORCA requires the API token be updated every so often,
  /// using the provided refresh token. This function will do that.
  void refreshTokens(bool force_refresh = false);

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

public:
  OrcaServerHelper() = default;
  virtual ~OrcaServerHelper() = default;

  /// @brief Return the name of this server helper, must be the
  /// same as the QPU configuration file.
  const std::string name() const override { return "orca"; }

  /// @brief Return the POST/GET required headers.
  /// @return
  RestHeaders getHeaders() override;

  /// @brief Set the server configuration.
  void initialize(BackendConfig config) override;

  /// @brief Create a job payload for the provided TBI parameters
  ServerJobPayload createJob(cudaq::orca::TBIParameters params);

  /// @brief Create a job payload for the provided quantum codes
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    std::vector<ServerMessage> jobs;
    ServerMessage job;
    jobs.push_back(job);

    std::map<std::string, std::string> headers;

    // Return a tuple containing the job path, headers, and the job message
    auto ret = std::make_tuple("", headers, jobs);
    return ret;
  };

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse) override;

  // /// @brief Given a completed job response, map back to the sample_result
  // sample_result processResults(ServerMessage &postJobResponse);

  /// @brief Given a completed job response, map back to the sample_result
  sample_result processResults(ServerMessage &postJobResponse,
                               std::string &jobID) override;
};

} // namespace cudaq
