/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "OrcaServerHelper.h"
#include "common/Future.h"
#include "common/Logger.h"
#include "common/Registry.h"
#include "orca_qpu.h"

namespace cudaq {

// Initialize the ORCA server helper with a given backend configuration
void OrcaServerHelper::initialize(BackendConfig config) {
  backendConfig = config;

  // Set the machine
  auto iter = backendConfig.find("machine");
  if (iter != backendConfig.end())
    machine = iter->second;

  // Set an alternate base URL if provided
  iter = backendConfig.find("url");
  if (iter != backendConfig.end()) {
    baseUrl = iter->second;
    if (!baseUrl.ends_with("/"))
      baseUrl += "/";
  }
}

// Create a job for the ORCA QPU
ServerJobPayload
OrcaServerHelper::createJob(cudaq::orca::TBIParameters params) {
  std::vector<ServerMessage> jobs;
  ServerMessage job;

  // Construct the job message
  job["target"] = machine;

  job["input_state"] = params.input_state;
  job["loop_lengths"] = params.loop_lengths;
  job["bs_angles"] = params.bs_angles;
  job["ps_angles"] = params.ps_angles;
  job["n_samples"] = params.n_samples;

  jobs.push_back(job);

  // Return a tuple containing the job path, headers, and the job message
  auto ret = std::make_tuple(baseUrl + "v1/submit", getHeaders(), jobs);
  return ret;
}

// ServerJobPayload
// OrcaServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
//   throw std::runtime_error("createJob on circuitCodes args not implemented");
// }

// Process the results from a job
sample_result OrcaServerHelper::processResults(ServerMessage &postJobResponse) {
  auto results = postJobResponse.at("results");

  CountsDictionary counts;
  // Process the results
  for (const auto &key : results) {
    counts[key] += 1;
  }

  // Create an execution result
  ExecutionResult executionResult(counts);
  // Return a sample result
  auto ret = sample_result(executionResult);
  return ret;
}

std::map<std::string, std::string>
OrcaServerHelper::generateRequestHeader() const {
  std::string apiKey, refreshKey, timeStr;
  std::map<std::string, std::string> headers{
      // {"Authorization", apiKey},
      {"Content-Type", "application/json"},
      // {"Connection", "keep-alive"},
      // {"Accept", "*/*"}
  };
  return headers;
}

// Get the headers for the API requests
RestHeaders OrcaServerHelper::getHeaders() { return generateRequestHeader(); }

// From a server message, extract the job ID
std::string OrcaServerHelper::extractJobId(ServerMessage &postResponse) {
  // If the response does not contain the key 'id', throw an exception
  if (!postResponse.contains("job_id"))
    throw std::runtime_error("ServerMessage doesn't contain 'job_id' key.");

  // Return the job ID from the response
  auto ret = postResponse.at("job_id");
  return ret;
}

std::string OrcaServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return baseUrl + "v1/get_job/" + extractJobId(postResponse);
}

std::string OrcaServerHelper::constructGetJobPath(std::string &jobId) {
  return baseUrl + "v1/get_job/" + jobId;
}

bool OrcaServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  auto status = getJobResponse["status"].get<std::string>();
  if (status == "failed") {
    std::string msg = "";
    if (getJobResponse.count("error"))
      msg = getJobResponse["error"]["text"].get<std::string>();
    throw std::runtime_error("Job failed to execute msg = [" + msg + "]");
  }

  return status == "completed";
}

} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::OrcaServerHelper::RegistryType)
CUDAQ_REGISTER_TYPE(cudaq::OrcaServerHelper, cudaq::OrcaServerHelper, orca)
