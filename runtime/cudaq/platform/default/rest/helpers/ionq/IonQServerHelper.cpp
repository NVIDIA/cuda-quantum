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
    backendConfig = config;

    // Set any other config you need...
    backendConfig["url"] = "https://api.ionq.co/v0.3";
    backendConfig["user_agent"] = "cudaq/0.3.0";
    backendConfig["target"] = "simulator";
    backendConfig["qubits"] = 29;
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

  std::vector<ServerMessage> jobs;
  for (const auto &circuitCode : circuitCodes) {
    ServerMessage job;
    job["target"] = backendConfig["target"];
    job["shots"] = static_cast<int>(shots);
    job["input"] = {{"format", "quil"}, {"quil", circuitCode.code}};
    jobs.push_back(job);
  }
  ServerMessage request;
  request["qubits"] = backendConfig["qubits"];
  request["shots"] = static_cast<int>(shots);
  request["job"] = jobs;
  return std::make_tuple("/jobs", getHeaders(),
                         std::vector<ServerMessage>{request});
}

std::string IonQServerHelper::extractJobId(ServerMessage &postResponse) {
  // return "JOB ID HERE, can extract from postResponse";
  return postResponse.at("id");
}

std::string IonQServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  // return "Get Job URL";
  return postResponse.at("output").at("uri"); // todo: use find to check keys
}

std::string IonQServerHelper::constructGetJobPath(std::string &jobId) {
  // return "Get Job URL from JOB ID string";
  return url + "/jobs?id=" + jobId;
}

bool IonQServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // return true if job is done, false otherwise
  return getJobResponse.at("status") == "completed"; // todo: use status enum
}

cudaq::sample_result
IonQServerHelper::processResults(ServerMessage &postJobResponse) {
  // results come back as results :{ "regName" : ['00','01',...], "regName2":
  // [...]} Map results back to a sample_result
  cudaq::sample_result result;
  auto data = postJobResponse.at("output").at("result");
  for (const auto &pair : data.items()) {
    std::vector<std::string> bits = pair.value();
    std::vector<double> probabilities;
    for (const auto &bit : bits) {
      probabilities.push_back(std::stod(bit) * std::stod(bit));
    }
    result[pair.key()] = {probabilities};
  }
  return result;
}

RestHeaders IonQServerHelper::getHeaders() {
  //   return  generateRequestHeader();
  RestHeaders headers;
  headers.insert({"Authorization", "apiKey " + backendConfig.at("token")});
  headers.insert({"Content-Type", "application/json"});
  headers.insert({"User-Agent", backendConfig.at("user_agent")});

  return headers;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IonQServerHelper, ionq)
