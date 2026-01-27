/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BraketServerHelper.h"
#include "common/FmtCore.h"

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
  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    CUDAQ_INFO("Emulation is enabled, ignore all Amazon Braket connection "
               "specific information.");
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

  if (outputNames.find(jobID) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobID);

  auto &output_names = outputNames[jobID];
  for (auto &[result, info] : output_names) {
    CUDAQ_INFO("Qubit {} Result {} Name {}", info.qubitNum, result,
               info.registerName);
  }

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

  // Full execution results include compiler-generated qubits, which are
  // undesirable to the user.
  cudaq::ExecutionResult fullExecResults{counts};
  auto fullSampleResults = cudaq::sample_result{fullExecResults};

  // clang-format off
  // The following code strips out and reorders the outputs based on output_names.
  // For example, if `counts` is something like:
  //      { 11111:62 01111:12 11110:12 01110:12 }
  // And if we want to discard the first bit (because qubit 0 was a
  // compiler-generated qubit), that maps to something like this:
  // -----------------------------------------------------
  // Qubit  Index - x1234    x1234    x1234    x1234
  // Result Index - x0123    x0123    x0123    x0123
  //              { 11111:62 01111:12 11110:12 01110:12 }
  //              { x1111:62 x1111:12 x1110:12 x1110:12 }
  //                  \--- v ---/       \--- v ---/
  //              {    1111:(62+12)     x1110:(12+12)   }
  //              {    1111:74           1110:24        }
  // -----------------------------------------------------
  // clang-format on

  std::vector<ExecutionResult> execResults;

  // Get a reduced list of qubit numbers that were in the original program
  // so that we can slice the output data and extract the bits that the user
  // was interested in. Sort by OpenQasm2 qubit number.
  std::vector<std::size_t> qubitNumbers;
  qubitNumbers.reserve(output_names.size());
  for (auto &[result, info] : output_names) {
    qubitNumbers.push_back(info.qubitNum);
  }

  // For each original counts entry in the full sample results, reduce it
  // down to the user component and add to userGlobal. If qubitNumbers is empty,
  // that means all qubits were measured.
  if (qubitNumbers.empty()) {
    execResults.emplace_back(ExecutionResult{fullSampleResults.to_map()});
  } else {
    auto subset = fullSampleResults.get_marginal(qubitNumbers);
    execResults.emplace_back(ExecutionResult{subset.to_map()});
  }

  // Return a sample result including the global register and all individual
  // registers.
  auto ret = cudaq::sample_result(execResults);
  return ret;
}

void BraketServerHelper::setOutputNames(const std::string &taskId,
                                        const std::string &output_names) {
  // Parse `output_names` into jobOutputNames.
  // Note: See `ExtendMeasurePattern` of `CombineMeasurements.cpp
  // for an example of how this was populated.
  OutputNamesType jobOutputNames;
  nlohmann::json outputNamesJSON = nlohmann::json::parse(output_names);
  for (const auto &el : outputNamesJSON[0]) {
    auto result = el[0].get<std::size_t>();
    auto qubitNum = el[1][0].get<std::size_t>();
    auto registerName = el[1][1].get<std::string>();
    jobOutputNames[result] = {qubitNum, registerName};
  }

  outputNames[taskId] = jobOutputNames;
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::BraketServerHelper, braket)
