/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BraketServerHelper.h"

namespace {
std::string prepareOpenQasm(std::string source) {
  const std::regex includeRE{"include \".*\";"};
  source = std::regex_replace(source, includeRE, "");
  const std::regex cxToCnot{"\\scx\\s"};
  source = std::regex_replace(source, cxToCnot, " cnot ");
  return source;
}

} // namespace

namespace cudaq {

std::string getDeviceArn(const std::string &machine) {
  if (machine.starts_with("arn:aws:braket")) {
    return machine;
  }

  if (deviceArns.contains(machine)) {
    return deviceArns.at(machine);
  }

  std::string knownMachines;
  for (const auto &machine : deviceArns)
    knownMachines += machine.first + " ";
  const auto errorMessage =
      fmt::format("Machine \"{}\" is invalid. Machine must be either a Braket "
                  "device ARN or one of the known devices: {}",
                  machine, knownMachines);
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
  cudaq::info("Initializing Amazon Braket backend.");

  // Fetch machine info before checking emulate because we want to be able to
  // emulate specific machines.
  auto machine = getValueOrDefault(config, "machine", SV1);
  auto deviceArn = getDeviceArn(machine);
  cudaq::info("Running on device {}", deviceArn);

  config["defaultBucket"] = getValueOrDefault(config, "default_bucket", "");
  config["deviceArn"] = deviceArn;
  config["qubits"] = deviceQubitCounts.contains(deviceArn)
                         ? deviceQubitCounts.at(deviceArn)
                         : DEFAULT_QUBIT_COUNT;
  if (!config["shots"].empty())
    this->setShots(std::stoul(config["shots"]));

  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    cudaq::info("Emulation is enabled, ignore all Braket connection specific "
                "information.");
    backendConfig = std::move(config);
    return;
  }

  parseConfigForCommonParams(config);

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

    taskRequest["qubits"] = backendConfig.at("qubits");
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

  cudaq::info("Created job payload for braket, language is OpenQASM 2.0, "
              "targeting device {}",
              backendConfig.at("deviceArn"));

  return ret;
};

sample_result BraketServerHelper::processResults(ServerMessage &resultsJson,
                                                 std::string &) {
  CountsDictionary counts;

  auto const &measurements = resultsJson.at("measurements");

  for (auto const &m : measurements) {
    std::string bitString = "";
    for (int bit : m) {
      bitString += std::to_string(bit);
    }
    counts[bitString]++;
  }

  return sample_result{ExecutionResult{counts}};
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::BraketServerHelper, braket)