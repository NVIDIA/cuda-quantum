/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/FmtCore.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/utils/cudaq_utils.h"
#include <bitset>
#include <fstream>
#include <map>
#include <thread>

namespace cudaq {

class QuakeServerHelper : public ServerHelper, public QirServerHelper {
  std::string url = "http://localhost:62451";

public:
  const std::string name() const override { return "quake_fake"; }

  RestHeaders getHeaders() override { return RestHeaders(); }

  void initialize(BackendConfig config) override {}

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    std::vector<ServerMessage> tasks;

    for (auto &circuitCode : circuitCodes) {
      ServerMessage message;
      message["ir"] = circuitCode.code;
      message["shots"] = shots;
      tasks.push_back(message);
    }

    return std::make_tuple(url + "/job", RestHeaders(), tasks);
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    return postResponse[0]["id"];
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // Return the job path
    return url + "/job/" + postResponse["id"].get<std::string>();
  }

  std::string extractOutputLog(ServerMessage &postJobResponse,
                               std::string &jobId) override {
    if (postJobResponse[0]["status"] == "error")
      throw std::runtime_error(
          std::string("Job failed with error: ") +
          postJobResponse[0]["message"].get<std::string>());

    return postJobResponse[0]["qir_output"];
  }

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override {
    return url + "/job/" + jobId;
  }

  bool jobIsDone(ServerMessage &getJobResponse) override { return true; }

  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override {
    return cudaq::sample_result();
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuakeServerHelper, quake_fake)
