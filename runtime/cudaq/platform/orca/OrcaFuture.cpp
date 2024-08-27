/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrcaFuture.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include <thread>

namespace cudaq::orca::details {

cudaq::sample_result Orcafuture::get() {
  if (wrapsFutureSampling)
    return inFuture.get();

  RestClient client;
  auto serverHelper = registry::get<ServerHelper>(qpuName);
  serverHelper->initialize(serverConfig);
  auto headers = serverHelper->getHeaders();

  std::vector<ExecutionResult> results;
  for (auto &id : jobs) {
    cudaq::info("Future retrieving results for {}.", id.first);

    auto jobGetPath = serverHelper->constructGetJobPath(id.first);

    cudaq::info("Future got job retrieval path as {}.", jobGetPath);
    auto resultResponse = client.get(jobGetPath, "", headers);
    while (!serverHelper->jobIsDone(resultResponse)) {
      auto polling_interval =
          serverHelper->nextResultPollingInterval(resultResponse);
      std::this_thread::sleep_for(polling_interval);
      resultResponse = client.get(jobGetPath, "", headers);
    }

    auto c = serverHelper->processResults(resultResponse, id.first);

    // If there are multiple jobs, this is likely a spin_op.
    // If so, use the job name instead of the global register.
    if (jobs.size() > 1) {
      results.emplace_back(c.to_map(), id.second);
      results.back().sequentialData = c.sequential_data();
    } else {
      // For each register, add the results into result.
      for (auto &regName : c.register_names()) {
        results.emplace_back(c.to_map(regName), regName);
        results.back().sequentialData = c.sequential_data(regName);
      }
    }
  }

  return cudaq::sample_result(results);
}

Orcafuture &Orcafuture::operator=(Orcafuture &other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  return *this;
}

Orcafuture &Orcafuture::operator=(Orcafuture &&other) {
  jobs = other.jobs;
  qpuName = other.qpuName;
  serverConfig = other.serverConfig;
  if (other.wrapsFutureSampling) {
    wrapsFutureSampling = true;
    inFuture = std::move(other.inFuture);
  }
  return *this;
}

std::ostream &operator<<(std::ostream &os, Orcafuture &f) {
  if (f.wrapsFutureSampling)
    throw std::runtime_error(
        "Cannot persist a Orcafuture for a local kernel execution.");

  nlohmann::json j;
  j["jobs"] = f.jobs;
  j["qpu"] = f.qpuName;
  j["config"] = f.serverConfig;
  os << j.dump(4);
  return os;
}

std::istream &operator>>(std::istream &is, Orcafuture &f) {
  nlohmann::json j;
  try {
    is >> j;
  } catch (...) {
    throw std::runtime_error(
        "Formatting error; could not parse input as json.");
  }
  f.jobs = j["jobs"].get<std::vector<Orcafuture::Job>>();
  f.qpuName = j["qpu"].get<std::string>();
  f.serverConfig = j["config"].get<std::map<std::string, std::string>>();
  return is;
}

} // namespace cudaq::orca::details
