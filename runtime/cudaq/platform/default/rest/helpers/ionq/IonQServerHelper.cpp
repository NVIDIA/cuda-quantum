/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include <fstream>
#include <iostream>
#include <thread>

namespace cudaq {

class IonQServerHelper : public ServerHelper {
protected:
  /// @brief Base URL for ionq api
  std::string url = "https://api.ionq.co/v0.3/"

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "ionq"; }
  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    backendConfig = config;

    // Set any other config you need..
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

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse) override;
};

ServerJobPayload
IonQServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {

  // Goal here is to build up the ServerPayload,
  // which is a tuple containing the Job Post URL, the
  // REST headers, and a vector of json messages containing the jobs to execute

  // return the job payload
  return ServerJobPayload{};
}

std::string IonQServerHelper::extractJobId(ServerMessage &postResponse) {
  // return "JOB ID HERE, can extract from postResponse";
  return postResponse["id"];
}

std::string IonQServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  // return "Get Job URL";
  return postResponse["output"]["uri"];
}

std::string IonQServerHelper::constructGetJobPath(std::string &jobId) {
  // return "Get Job URL from JOB ID string";
  return url + "jobs?id=" + jobId;
}

bool IonQServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // return true if job is done, false otherwise
  return getJobResponse["status"] == "completed"; // todo: use status enum
}

cudaq::sample_result
IonQServerHelper::processResults(ServerMessage &postJobResponse) {
  // results come back as results :{ "regName" : ['00','01',...], "regName2":
  // [...]}
  // Map results back to a sample_result,
  // here's an example
  //   auto results = postJobResponse["results"];
  //   std::vector<ExecutionResult> srs;
  //   for (auto &result : results.items()) {
  //     cudaq::CountsDictionary counts;
  //     auto regName = result.key();
  //     auto bitResults = result.value().get<std::vector<std::string>>();
  //     for (auto &bitResult : bitResults) {
  //       if (counts.count(bitResult))
  //         counts[bitResult]++;
  //       else
  //         counts.insert({bitResult, 1});
  //     }

  //     srs.emplace_back(counts);
  //   }
  //   return sample_result(srs);

  return sample_result();
}

RestHeaders IonQServerHelper::getHeaders() {
  //   return  generateRequestHeader();
  return RestHeaders();
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
