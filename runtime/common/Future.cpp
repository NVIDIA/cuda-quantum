/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Future.h"
#include "Logger.h"
#include "ObserveResult.h"
#include "RestClient.h"
#include "ServerHelper.h"
#include <thread>

namespace cudaq::details {

sample_result future::get() {
  if (wrapsFutureSampling)
    return inFuture.get();

#ifdef CUDAQ_RESTCLIENT_AVAILABLE
  RestClient client;
  auto serverHelper = registry::get<ServerHelper>(qpuName);
  serverHelper->initialize(serverConfig);
  auto headers = serverHelper->getHeaders();
  std::vector<ExecutionResult> results;
  for (auto &id : jobs) {
    CUDAQ_INFO("Future retrieving results for {}.", id.first);

    auto jobGetPath = serverHelper->constructGetJobPath(id.first);

    CUDAQ_INFO("Future got job retrieval path as {}.", jobGetPath);
    auto resultResponse =
        client.get(jobGetPath, "", headers, false, serverHelper->getCookies());
    while (!serverHelper->jobIsDone(resultResponse)) {
      auto polling_interval =
          serverHelper->nextResultPollingInterval(resultResponse);
      std::this_thread::sleep_for(polling_interval);
      resultResponse = client.get(jobGetPath, "", headers, false,
                                  serverHelper->getCookies());
    }

    if (resultType == ExecutionContextType::run) {
      QirServerHelper *qirServerHelper =
          dynamic_cast<QirServerHelper *>(serverHelper.get());
      if (!qirServerHelper)
        throw std::runtime_error("To support `run` API, " + qpuName +
                                 " must inherit `QirServerHelper` class");
      if (!inFutureRawOutput)
        throw std::runtime_error(
            "cudaq::details::future::get() for 'run' requires a raw output "
            "pointer but it was not provided.");

      const auto qirOutputLog =
          qirServerHelper->extractOutputLog(resultResponse, id.first);
      inFutureRawOutput->assign(qirOutputLog.begin(), qirOutputLog.end());
      return sample_result();
    }

    auto c = serverHelper->processResults(resultResponse, id.first);
    if (isObserve()) {
      // Use the job name instead of the global register.
      results.emplace_back(c.to_map(), id.second);
      results.back().sequentialData = c.sequential_data();
    } else {
      if (c.has_expectation()) {
        // If the QPU returns the data with expectation values, just use it
        // directly.
        // This can be the case for remote emulation/simulation providers who
        // compute the expectation value for us.
        return c;
      }

      // For each register, add the results into result.
      for (auto &regName : c.register_names()) {
        results.emplace_back(c.to_map(regName), regName);
        results.back().sequentialData = c.sequential_data(regName);
      }
    }
  }

  return sample_result(results);
#else
  throw std::runtime_error("cudaq::details::future::get() requires REST Client "
                           "but CUDA-Q was built without it.");
  return sample_result();
#endif
}

future &future::operator=(future &other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  resultType = other.resultType;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  inFutureRawOutput = other.inFutureRawOutput;
  return *this;
}

future &future::operator=(future &&other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  resultType = other.resultType;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  inFutureRawOutput = other.inFutureRawOutput;
  return *this;
}

std::ostream &operator<<(std::ostream &os, future &f) {
  if (f.wrapsFutureSampling)
    throw std::runtime_error(
        "Cannot persist a cudaq::future for a local kernel execution.");

  nlohmann::json j;
  j["jobs"] = f.jobs;
  j["qpu"] = f.qpuName;
  j["config"] = f.serverConfig;
  j["resultType"] = f.resultType;
  os << j.dump(4);
  return os;
}

std::istream &operator>>(std::istream &is, future &f) {
  nlohmann::json j;
  try {
    is >> j;
  } catch (...) {
    throw std::runtime_error(
        "Formatting error; could not parse input as json.");
  }
  f.jobs = j["jobs"].get<std::vector<future::Job>>();
  f.qpuName = j["qpu"].get<std::string>();
  f.serverConfig = j["config"].get<std::map<std::string, std::string>>();
  f.resultType = j["resultType"].get<ExecutionContextType>();
  return is;
}

} // namespace cudaq::details
