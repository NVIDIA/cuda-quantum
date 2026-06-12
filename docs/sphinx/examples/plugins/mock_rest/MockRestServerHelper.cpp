/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ServerHelper.h"
#include "nlohmann/json.hpp"
#include <string>

namespace cudaq {

class MockRestServerHelper : public ServerHelper {
  std::string url = "http://localhost:62454";

public:
  const std::string name() const override { return "mock_rest"; }

  void initialize(BackendConfig config) override {
    backendConfig = std::move(config);
    parseConfigForCommonParams(backendConfig);
    if (auto iter = backendConfig.find("url"); iter != backendConfig.end())
      url = iter->second;
  }

  RestHeaders getHeaders() override { return {}; }

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    std::vector<ServerMessage> tasks;
    tasks.reserve(circuitCodes.size());
    for (const auto &circuitCode : circuitCodes) {
      ServerMessage task;
      task["kernel"] = circuitCode.name;
      task["code"] = circuitCode.code;
      task["shots"] = shots;
      tasks.push_back(std::move(task));
    }

    return std::make_tuple(url + "/jobs", RestHeaders(), std::move(tasks));
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    if (postResponse.is_array() && !postResponse.empty())
      return postResponse.front().value("id", "mock-rest-job");
    return postResponse.value("id", "mock-rest-job");
  }

  std::string constructGetJobPath(std::string &jobId) override {
    return url + "/jobs/" + jobId;
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    auto jobId = extractJobId(postResponse);
    return constructGetJobPath(jobId);
  }

  bool jobIsDone(ServerMessage &getJobResponse) override {
    const auto status = getJobResponse.value("status", "done");
    return status == "done" || status == "completed" || status == "succeeded";
  }

  sample_result processResults(ServerMessage &postJobResponse,
                               std::string &jobId) override {
    CountsDictionary counts;
    if (postJobResponse.contains("counts")) {
      for (auto &[bits, count] : postJobResponse["counts"].items())
        counts[bits] = count.get<std::size_t>();
    } else {
      counts["0"] = shots;
    }

    return sample_result(ExecutionResult(counts));
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::MockRestServerHelper, mock_rest)
