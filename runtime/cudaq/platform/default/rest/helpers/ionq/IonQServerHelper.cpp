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
public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "ionq"; }
  RestHeaders getHeaders() override;

  void initialize(BackendConfig config) override {
    std::cout << "IonQ Initialized" << std::endl;
    backendConfig = config;

    // Set any other config you need...
    backendConfig["url"] = "https://api.ionq.co";
    backendConfig["version"] = "v0.3";
    backendConfig["user_agent"] = "cudaq/0.3.0";
    backendConfig["target"] = "simulator";
    backendConfig["qubits"] = 29;
    backendConfig["token"] = "giveme403";

    backendConfig["job_path"] =
        backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";
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
  std::cout << "Creating Job" << std::endl;

  std::vector<ServerMessage> jobs;
  for (const auto &circuitCode : circuitCodes) {
    ServerMessage job;
    job["target"] = backendConfig.at("target");
    job["shots"] = static_cast<int>(shots);
    job["input"] = {{"format", "qir"}, {"data", circuitCode.code}};
    jobs.push_back(job);
  }
  ServerMessage request;
  request["qubits"] = backendConfig.at("qubits");
  request["shots"] = static_cast<int>(shots);
  request["job"] = jobs;

  return std::make_tuple(backendConfig.at("job_path"), getHeaders(),
                         std::vector<ServerMessage>{request});
}

std::string IonQServerHelper::extractJobId(ServerMessage &postResponse) {
  // return "JOB ID HERE, can extract from postResponse";
  std::cout << "Extracting Job ID" << std::endl;
  return postResponse.at("id");
}

std::string IonQServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  // return "Get Job URL";
  std::cout << postResponse << std::endl;
  return backendConfig.at("url") +
         postResponse.at("results_url"); // todo: use find to check keys
}

std::string IonQServerHelper::constructGetJobPath(std::string &jobId) {
  // return "Get Job URL from JOB ID string";
  std::cout << jobId << std::endl;
  return backendConfig.at("job_path") + "?id=" + jobId;
}

bool IonQServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // return true if job is done, false otherwise
  std::cout << getJobResponse << std::endl;
  return getJobResponse.at("status") == "completed"; // todo: use status enum
}

cudaq::sample_result
IonQServerHelper::processResults(ServerMessage &postJobResponse) {
  // results come back as results :{ "regName" : ['00','01',...], "regName2":
  // [...]}
  // Map results back to a sample_result,
  // here's an example
  std::cout << postJobResponse << std::endl;
  auto results = postJobResponse["results"];
  std::vector<ExecutionResult> srs;
  for (auto &result : results.items()) {
    cudaq::CountsDictionary counts;
    auto regName = result.key();
    auto bitResults = result.value().get<std::vector<std::string>>();
    for (auto &bitResult : bitResults) {
      if (counts.count(bitResult))
        counts[bitResult]++;
      else
        counts.insert({bitResult, 1});
    }
    srs.emplace_back(counts);
  }
  return sample_result(srs);
}

RestHeaders IonQServerHelper::getHeaders() {
  //   return  generateRequestHeader();
  std::cout << "Getting Request Headers" << std::endl;

  RestHeaders headers;
  headers["Authorization"] = "apiKey " + backendConfig.at("token");
  headers["Content-Type"] = "application/json";
  headers["User-Agent"] = backendConfig.at("user_agent");
  return headers;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
