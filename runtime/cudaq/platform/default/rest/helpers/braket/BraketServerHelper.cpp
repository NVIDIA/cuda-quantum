/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BraketServerHelper.h"
#include "common/FmtCore.h"
#include "nlohmann/json.hpp"

namespace {
std::string prepareOpenQasm(std::string source) {
  const std::regex includeRE{"include \".*\";"};
  source = std::regex_replace(source, includeRE, "");
  const std::regex cxToCnot{"\\scx\\s"};
  source = std::regex_replace(source, cxToCnot, " cnot ");
  const std::regex ccxToCcnot{"\\sccx\\s"};
  source = std::regex_replace(source, ccxToCcnot, " ccnot ");
  return source;
}

} // namespace

namespace cudaq {

std::string checkDeviceArn(const std::string &machine) {
  if (machine.starts_with("arn:aws:braket"))
    return machine;
  const auto errorMessage = fmt::format("Machine \"{}\" is invalid. Machine "
                                        "must be an Amazon Braket device ARN.",
                                        machine);
  throw std::runtime_error(errorMessage);
}

// Implementation of the getValueOrDefault function
std::string
BraketServerHelper::getValueOrDefault(const BackendConfig &config,
                                      const std::string &key,
                                      const std::string &defaultValue) const {
  return config.find(key) != config.end() ? config.at(key) : defaultValue;
}

// Initialize the Braket server helper with a given backend configuration
void BraketServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing Amazon Braket backend.");
  // Fetch machine info before checking emulate because we want to be able to
  // emulate specific machines, defaults to state vector simulator
  auto machine =
      getValueOrDefault(config, "machine",
                        "arn:aws:braket:::device/quantum-simulator/amazon/sv1");
  auto deviceArn = checkDeviceArn(machine);
  CUDAQ_INFO("Running on device {}", deviceArn);
  config["defaultBucket"] = getValueOrDefault(config, "default_bucket", "");
  config["deviceArn"] = deviceArn;
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));
  parseConfigForCommonParams(config);
  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    CUDAQ_INFO("Emulation is enabled, ignore all Amazon Braket connection "
               "specific information.");
    backendConfig = std::move(config);
    return;
  }
  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
};

// Create a job for Braket
ServerJobPayload
BraketServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  ServerJobPayload ret;
  std::vector<ServerMessage> &tasks = std::get<2>(ret);
  for (auto &circuitCode : circuitCodes) {
    // Construct the job message
    ServerMessage taskRequest;
    taskRequest["name"] = circuitCode.name;
    taskRequest["deviceArn"] = backendConfig.at("deviceArn");
    taskRequest["input"]["format"] = "qasm2";
    taskRequest["input"]["data"] = circuitCode.code;
    auto action = nlohmann::json::parse(
        "{\"braketSchemaHeader\": {\"name\": \"braket.ir.openqasm.program\", "
        "\"version\": \"1\"}, \"source\": \"\", \"inputs\": {}}");
    action["source"] = prepareOpenQasm(circuitCode.code);
    taskRequest["action"] = action.dump();
    taskRequest["shots"] = shots;
    tasks.push_back(taskRequest);
  }
  CUDAQ_INFO("Created job payload for braket, language is OpenQASM 2.0, "
             "targeting device {}",
             backendConfig.at("deviceArn"));
  return ret;
};

sample_result BraketServerHelper::processResults(ServerMessage &resultsJson,
                                                 std::string &jobID) {

  CountsDictionary counts;

  if (resultsJson.contains("measurements")) {
    auto const &measurements = resultsJson.at("measurements");

    for (auto const &m : measurements) {
      std::string bitString = "";
      for (int bit : m) {
        bitString += std::to_string(bit);
      }
      counts[bitString]++;
    }
  } else {
    auto const &probs = resultsJson.at("measurementProbabilities");
    int shots = resultsJson.at("taskMetadata").at("shots");
    for (auto const &measurementP : probs.items()) {
      std::string bitString = measurementP.key();
      double p = measurementP.value();
      counts[bitString] = std::round(p * shots);
    }
  }

  // Reconstruct the user-visible result order and named registers from the
  // enriched output_names. When no output_names exist for this job, return the
  // raw global register unchanged.
  if (auto result = tryReconstructFromDeviceIndexedCounts(jobID, counts))
    return *result;

  return cudaq::sample_result{cudaq::ExecutionResult{counts}};
}

void BraketServerHelper::setOutputNames(const std::string &taskId,
                                        const std::string &output_names) {
  // Parse enriched `output_names` into jobOutputNames. Each output-location
  // tuple is [qubitNum, registerName, outputPosition]; an old compiler omits
  // the third element, in which case fall back to the result index. The taskId
  // is the task ARN assigned by Braket at submission time. processResults is
  // always called with the same ARN (see BraketExecutor.cpp), so the key used
  // here and the jobID key used in tryReconstructFromDeviceIndexedCounts match.
  OutputNamesType jobOutputNames;
  nlohmann::json outputNamesJSON = nlohmann::json::parse(output_names);
  std::size_t resultIndex = 0;
  for (const auto &el : outputNamesJSON[0]) {
    auto result = el[0].get<std::size_t>();
    const auto &outputLocation = el[1];
    auto qubitNum = outputLocation[0].get<std::size_t>();
    auto registerName = outputLocation[1].get<std::string>();
    std::size_t outputPosition = resultIndex;
    if (outputLocation.size() > 2)
      outputPosition = outputLocation[2].get<std::size_t>();
    jobOutputNames[result] = {qubitNum, registerName, outputPosition};
    ++resultIndex;
  }

  outputNames[taskId] = jobOutputNames;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::BraketServerHelper, braket)
