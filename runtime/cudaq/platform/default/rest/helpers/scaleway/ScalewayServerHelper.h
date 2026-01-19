/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "qaas/Qaas.h"
#include "common/Logger.h"
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
  void
  initialize(BackendConfig config) override;

  RestHeaders
  getHeaders() override;

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  std::string
  extractJobId(ServerMessage &postResponse) override;

  bool
  jobIsDone(ServerMessage &getJobResponse) override;

  cudaq::sample_result
  processResults(ServerMessage &postJobResponse,
                std::string &jobId) override;

  std::string
  constructGetJobPath(std::string &jobId) override;

  std::string
  constructGetJobPath(ServerMessage &postResponse) override;

  virtual std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) overide {
    return std::chrono::microseconds(100);
  }

protected:
  void ensureSessionIsActive();
  std::string serializeKernelToQio(const std::string& code, size_t shots);
  std::string createModel(const std::string& name, const std::string& content)
private:
  const std::string m_defaultPlatformName = "EMU-CUDAQ-H100";
  cudaq::qaas::v1alpha1::V1Alpha1Client m_qaasClient;
  std::string m_targetPlatformName = "";
  std::string m_sessionId = "";
  std::string m_sessionDeduplicationId = "";
  std::string m_sessionName = "";
  std::string m_sessionMaxDuration = "";
  std::string m_sessionMaxIdleDuration = "";
  size_t m_shots;
};

} // namespace cudaq
