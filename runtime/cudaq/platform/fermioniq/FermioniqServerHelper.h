/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/utils/cudaq_utils.h"
#include <bitset>
#include <fstream>
#include <map>
#include <thread>

namespace cudaq {

/// @brief The FermioniqServerHelper class extends the ServerHelper class to
/// handle interactions with the Fermioniq server for submitting and retrieving
/// quantum computation jobs.
class FermioniqServerHelper : public ServerHelper {

  static constexpr int POLLING_INTERVAL_IN_SECONDS = 1;

  static constexpr const char *DEFAULT_URL =
      "https://fermioniq-api-fapp-prod.azurewebsites.net";
  static constexpr const char *DEFAULT_API_KEY =
      "gCUVmJOKVCdPKRYpgk7nNWM_kTAsZfPeYTbte2sNuKtXAzFuYdj9ag==";

  static constexpr const char *CFG_URL_KEY = "base_url";
  static constexpr const char *CFG_ACCESS_TOKEN_ID_KEY = "access_token_id";
  static constexpr const char *CFG_ACCESS_TOKEN_SECRET_KEY =
      "access_token_secret";
  static constexpr const char *CFG_API_KEY_KEY = "api_key";
  static constexpr const char *CFG_USER_AGENT_KEY = "user_agent";
  static constexpr const char *CFG_TOKEN_KEY = "token";

  static constexpr const char *CFG_REMOTE_CONFIG_KEY = "remote_config";
  static constexpr const char *CFG_NOISE_MODEL_KEY = "noise_model";
  static constexpr const char *CFG_BOND_DIM_KEY = "bond_dim";
  static constexpr const char *CFG_PROJECT_ID_KEY = "project_id";

public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "fermioniq"; }

  /// @brief Returns the headers for the server requests.
  RestHeaders getHeaders() override;

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Extracts the job ID from the server's response to a job submission.
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on the server's
  /// response to a job submission.
  std::string constructGetJobPath(ServerMessage &postResponse) override;

  /// @brief Constructs the URL for retrieving a job based on a job ID.
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;

  void refreshTokens(bool force_refresh);

#if 0
  /// @brief Update `passPipeline` with architecture-specific pass options
  void updatePassPipeline(const std::filesystem::path &platformPath,
                          std::string &passPipeline) override;
#endif

  /// @brief Return next results polling interval
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

private:
  /// @brief RestClient used for HTTP requests.
  // RestClient client;

  /// @brief API Key for Fermioniq API
  std::string token;

  /// @brief user_id of logged in user
  std::string userId;

  std::vector<std::string> circuit_names;

  /// @brief exp time of token
  std::chrono::_V2::system_clock::time_point tokenExpTime;

  /// @brief Helper method to retrieve the value of an environment variable.
  std::string getEnvVar(const std::string &key, const std::string &defaultVal,
                        const bool isRequired) const;

  /// @brief Helper function to get value from config or return a default value.
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const;

  /// @brief Helper method to check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;
};

} // namespace cudaq