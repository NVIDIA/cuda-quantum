/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>

namespace cudaq {

/// @brief The ScalewayServerHelper class extends the ServerHelper class to handle
/// interactions with the Amazon Scaleway server for submitting and retrieving
/// quantum computation jobs.
class ScalewayServerHelper : public ServerHelper {
public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "scaleway"; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  RestHeaders getHeaders() override { return {}; }

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  std::string extractJobId(ServerMessage &postResponse) override { return ""; }

  std::string constructGetJobPath(std::string &jobId) override { return ""; }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    return "";
  }

  bool jobIsDone(ServerMessage &getJobResponse) override { return true; }

  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;

  void setOutputNames(const std::string &taskId,
                      const std::string &output_names);

protected:
  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;
  RestHeaders generateRequestHeader(std::string) const;
  /// @brief Helper function to get value from the configuration or return a
  /// default value.
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const;

  void ensureSessionIsActive();
  std::string serializeKernelToQio(const std::string& code, size_t shots);
  std::string createModel(const std::string& name, const std::string& content)
private:
  RestClient m_client;
  std::string m_baseUrl = "https://api.scaleway.com/qaas/v1alpha1";
  std::string m_basePlatformName = "EMU-CUDAQ-H100";
  std::string m_projectId = "";
  std::string m_sessionId = "";
  std::string m_secretKey = "";
};

} // namespace cudaq
