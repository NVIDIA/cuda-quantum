/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/ExtraPayloadProvider.h"
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

class HorizonServerHelper : public ServerHelper {
  std::string url = "http://localhost:62450";

public:
  const std::string name() const override { return "horizon"; }

  RestHeaders getHeaders() override { return RestHeaders(); }

  void initialize(BackendConfig config) override {}

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    ServerMessage job;
    job["name"] = "test_job";

    const auto iter =
        runtimeTarget.runtimeConfig.find("extra_payload_provider");
    if (iter == runtimeTarget.runtimeConfig.end())
      throw std::runtime_error("No extra payload provider was set.");
    const auto extraPayloadProviderName = iter->second;

    auto &extraProviders = cudaq::getExtraPayloadProviders();
    const auto it = std::find_if(
        extraProviders.begin(), extraProviders.end(), [&](const auto &entry) {
          return entry->name() == extraPayloadProviderName;
        });
    if (it == extraProviders.end())
      throw std::runtime_error("ExtraPayloadProvider with name " +
                               extraPayloadProviderName + " not found.");

    auto *extraPayloadProvider = it->get();

    if (extraPayloadProvider->getPayloadType() != "test_type")
      throw std::runtime_error(
          fmt::format("Unsupported extra payload type: {}",
                      extraPayloadProvider->getPayloadType()));

    const std::string config =
        extraPayloadProvider->getExtraPayload(runtimeTarget);
    job["foo"]["bar"] = config;

    return std::make_tuple(url + "/job", RestHeaders(),
                           std::vector<ServerMessage>{job});
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    return "1234";
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // Return the job path
    return url + "/job/1234";
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

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::HorizonServerHelper, horizon)
