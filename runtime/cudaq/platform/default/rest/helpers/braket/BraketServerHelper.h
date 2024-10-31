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
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>

namespace cudaq {

const std::string SV1 = "sv1";
const std::string DM1 = "dm1";
const std::string TN1 = "tn1";
const std::string ARIA1 = "aria1";
const std::string ARIA2 = "aria2";
const std::string GARNET = "garnet";
const std::string AQUILA = "aquila";

const std::string SV1_ARN =
    "arn:aws:braket:::device/quantum-simulator/amazon/sv1";
const std::string DM1_ARN =
    "arn:aws:braket:::device/quantum-simulator/amazon/dm1";
const std::string TN1_ARN =
    "arn:aws:braket:::device/quantum-simulator/amazon/tn1";
const std::string ARIA1_ARN =
    "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1";
const std::string ARIA2_ARN =
    "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-2";
const std::string GARNET_ARN =
    "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet";
const std::string AQUILA_ARN =
    "arn:aws:braket:us-east-1::device/qpu/quera/Aquila";

const std::map<std::string, std::string> deviceArns = {
    {SV1, SV1_ARN},      {DM1, DM1_ARN},     {TN1, TN1_ARN},
    {ARIA1, ARIA1_ARN},  {ARIA2, ARIA2_ARN}, {GARNET, GARNET_ARN},
    {AQUILA, AQUILA_ARN}};

const std::map<std::string, uint> deviceQubitCounts = {
    {SV1_ARN, 34},   {DM1_ARN, 17},    {TN1_ARN, 50},    {ARIA1_ARN, 25},
    {ARIA2_ARN, 25}, {GARNET_ARN, 20}, {AQUILA_ARN, 256}};
const uint DEFAULT_QUBIT_COUNT = 50;

/// @brief The BraketServerHelper class extends the ServerHelper class to handle
/// interactions with the Amazon Braket server for submitting and retrieving
/// quantum computation jobs.
class BraketServerHelper : public ServerHelper {
public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "braket"; }

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
                                      std::string &jobId) override {
    return {};
  }

private:
  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;
  RestHeaders generateRequestHeader(std::string) const;
  /// @brief Helper function to get value from config or return a default value.
  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const;
};

} // namespace cudaq