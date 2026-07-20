/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraServerHelper.h"
#include "common/AnalogHamiltonian.h"

namespace cudaq {

void QuEraServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing QuEra via Amazon Braket.");
  // Hard-coded for now
  auto deviceArn = "arn:aws:braket:us-east-1::device/qpu/quera/Aquila";
  CUDAQ_INFO("Running on device {}", deviceArn);
  config["defaultBucket"] = getValueOrDefault(config, "default_bucket", "");
  config["deviceArn"] = deviceArn;
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));
  parseConfigForCommonParams(config);
  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Create a job for the QuEra quantum computer
ServerJobPayload
QuEraServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  ServerJobPayload ret;
  std::vector<ServerMessage> &tasks = std::get<2>(ret);
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage taskRequest;

    taskRequest["name"] = circuitCode.name;
    taskRequest["deviceArn"] = backendConfig.at("deviceArn");

    auto action = nlohmann::json::parse(circuitCode.code);
    action["braketSchemaHeader"]["name"] = "braket.ir.ahs.program";
    action["braketSchemaHeader"]["version"] = "1";
    taskRequest["action"] = action.dump();
    taskRequest["shots"] = shots;

    tasks.push_back(taskRequest);
  }
  CUDAQ_INFO("Created job payload for QuEra, targeting device {}",
             backendConfig.at("deviceArn"));
  return ret;
}

sample_result QuEraServerHelper::processResults(ServerMessage &resultsJson,
                                                std::string &) {
  std::vector<ExecutionResult> results;
  CountsDictionary globalReg;
  CountsDictionary preSeqReg;
  CountsDictionary postSeqReg;
  if (resultsJson.contains("measurements")) {
    auto const &measurements = resultsJson.at("measurements");
    for (auto const &m : measurements) {
      cudaq::ahs::ShotMeasurement sm = m;
      if (sm.shotMetadata.shotStatus == "Success") {
        auto pre = sm.shotResult.preSequence.value();
        std::string preString = "";
        for (int bit : pre)
          preString += std::to_string(bit);
        preSeqReg[preString]++;
        auto post = sm.shotResult.postSequence.value();
        std::string postString = "";
        for (int bit : post)
          postString += std::to_string(bit);
        postSeqReg[postString]++;
        /// Convert the AHS results to sampling
        /// Ref:
        /// https://docs.aws.amazon.com/braket/latest/developerguide/braket-get-started-hello-ahs.html#braket-get-started-analyzing-simulator-results
        std::vector<int> state_idx(pre.size());
        for (size_t i = 0; i < pre.size(); ++i)
          state_idx[i] = pre[i] * (1 + post[i]);
        std::string bitString = "";
        for (int bit : state_idx)
          bitString += std::to_string(bit);
        globalReg[bitString]++;
      }
    }
  }
  results.emplace_back(preSeqReg, "pre_sequence");
  results.emplace_back(postSeqReg, "post_sequence");
  results.emplace_back(globalReg, GlobalRegisterName);
  sample_result sampleResult(results);
  return sampleResult;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QuEraServerHelper, quera)
