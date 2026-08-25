/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"

namespace cudaq {

class PasqalServerHelper : public ServerHelper {
protected:
  /// @brief Server helper implementation for communicating with the REST API of
  /// Pasqal's cloud platform.
  const std::string baseUrl = "https://apis.pasqal.cloud";
  const std::string apiPath = "/core-fast/api";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "pasqal"; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  /// @brief Return the POST/GET required headers.
  /// @return
  RestHeaders getHeaders() override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Extract the job id from the server response from posting the job.
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Get the specific path required to retrieve job results. Construct
  /// specifically from the job id.
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Get the specific path required to retrieve job results. Construct
  /// from the full server response message.
  std::string constructGetJobPath(ServerMessage &postResponse) override;

  /// @brief Get the jobs results polling interval.
  /// @return
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(1);
  }

  /// @brief Return true if the job is done.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Given a successful job and the success response,
  /// retrieve the results and map them to a sample_result.
  /// @param postJobResponse
  /// @param jobId
  /// @return
  sample_result processResults(ServerMessage &postJobResponse,
                               std::string &jobId) override;
};

} // namespace cudaq
