/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"

#include "nlohmann/json.hpp"

#include <fstream>
#include <regex>
#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <unordered_set>

namespace cudaq {

class IQMServerHelper : public ServerHelper {

  struct qubitOrder {
    // Lightweight comparison for sorting strings ending in a number in
    // natural order. This assumes that all strings have either none or
    // the same prefix and there is a number. No checks on the string
    // composition is done for performance reasons.
    bool operator()(const std::string &a, const std::string &b) const {
      if (a.size() < b.size())
        return true;
      if (a.size() > b.size())
        return false;
      return a.compare(b) < 0;
    }
  };

protected:
  /// @brief The base URL
  std::string iqmServerUrl = "http://localhost/";

  /// @brief Authentication token
  std::optional<std::string> authToken = std::nullopt;

  /// @brief The default cortex-cli tokens file path
  std::optional<std::string> tokensFilePath = std::nullopt;

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Parse cortex-cli tokens JSON for the API access token
  std::optional<std::string> readApiToken() const {
    if (!tokensFilePath.has_value()) {
      cudaq::info(
          "tokensFilePath is not set, assuming no authentication is required");
      return std::nullopt;
    }

    std::string unwrappedTokensFilePath = tokensFilePath.value();
    std::ifstream tokensFile(unwrappedTokensFilePath);
    if (!tokensFile.is_open()) {
      throw std::runtime_error("Unable to open tokens file: " +
                               unwrappedTokensFilePath);
    }
    nlohmann::json tokens;
    tokensFile >> tokens;
    tokensFile.close();

    if (!tokens.count("access_token")) {
      throw std::runtime_error("No 'access_token' found in tokens file: " +
                               unwrappedTokensFilePath);
    }
    return tokens["access_token"].get<std::string>();
  }

  /// @brief Lookup table for translating the qubit names to index numbers
  std::map<std::string, uint, qubitOrder> qubitNameMap;

  /// @brief Adjacency map for each qubit
  std::vector<std::set<uint>> qubitAdjacencyMap;

  /// @brief full path+name of the file containing the quantum architecture
  std::string quantumArchitectureFilePath;

  /// @brief Fetch the quantum architecture from server
  void fetchQuantumArchitecture();

  /// @brief Write the dynamic quantum architecture file
  std::string writeQuantumArchitectureFile(void);

public:
  /// @brief Return the name of this server helper, must be the
  /// same as the qpu config file.
  const std::string name() const override { return "iqm"; }

  RestHeaders getHeaders() override { return generateRequestHeader(); }

  void initialize(BackendConfig config) override;

  /// @brief Create a job payload for the provided quantum codes
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Return the job id from the previous job post
  std::string extractJobId(ServerMessage &postResponse) override;

  /// @brief Return the URL for retrieving job results
  std::string constructGetJobPath(ServerMessage &postResponse) override;
  std::string constructGetJobPath(std::string &jobId) override;

  /// @brief Return next results polling interval
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

  /// @brief Return true if the job is done
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;

  /// @brief Update `passPipeline` with architecture-specific pass options
  void updatePassPipeline(const std::filesystem::path &platformPath,
                          std::string &passPipeline) override;

  /// @brief Default destructor removes the dynamic quantum architecture file
  ~IQMServerHelper() {
    if (!quantumArchitectureFilePath.empty()) {
      if (unlink(quantumArchitectureFilePath.c_str()) != 0) {
        cudaq::info("Failed to delete {} with errno:{}",
                    quantumArchitectureFilePath, errno);
      }
    }
  }
};

void IQMServerHelper::initialize(BackendConfig config) {
  backendConfig = config;

  bool emulate = false;
  auto iter = backendConfig.find("emulate");
  if (iter != backendConfig.end()) {
    emulate = iter->second == "true";
  }

  // Set an alternate base URL if provided.
  iter = backendConfig.find("url");
  if (iter != backendConfig.end()) {
    iqmServerUrl = iter->second;
  }

  // Allow overriding IQM Server Url, the compiled program will still work if
  // architecture matches. This is useful in case we're using the same program
  // against different backends, for example simulated and actually connected
  // to the hardware.
  auto envIqmServerUrl = getenv("IQM_SERVER_URL");
  if (envIqmServerUrl) {
    iqmServerUrl = std::string(envIqmServerUrl);
  }

  if (!iqmServerUrl.ends_with("/"))
    iqmServerUrl += "/";
  cudaq::debug("iqmServerUrl = {}", iqmServerUrl);

  if (emulate) {
    cudaq::info("Emulation is enabled, ignore tokens file and IQM Server URL");
    return;
  }

  auto token = getenv("IQM_TOKEN");
  if (token) {
    authToken = std::string(token);
    cudaq::debug("Using authorization token from environment variable");
  } else {
    // Set alternative iqmclient-cli tokens file path if provided via env var
    auto envTokenFilePath = getenv("IQM_TOKENS_FILE");
    auto defaultTokensFilePath =
        std::string(getenv("HOME")) + "/.cache/iqm-client-cli/tokens.json";
    if (envTokenFilePath) {
      tokensFilePath = std::string(envTokenFilePath);
    } else if (cudaq::fileExists(defaultTokensFilePath)) {
      tokensFilePath = defaultTokensFilePath;
      cudaq::debug("Setting default path for tokens file");
    }
    cudaq::debug("tokensFilePath = {}", tokensFilePath.value_or("not set"));
  }
}

ServerJobPayload
IQMServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;

  // cuda-quantum expects every circuit to be a separate job,
  // so we cannot use the batch mode
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message = ServerMessage::object();
    message["qubit_mapping"] = ServerMessage::array();
    message["circuits"] = ServerMessage::array();
    message["shots"] = shots;

    // Apply the mapping derived from the dynamic quantum architecture.
    for (auto &[key, value] : qubitNameMap) {
      nlohmann::json singleQubitMapping;
      singleQubitMapping["logical_name"] = "QB" + std::to_string(value + 1);
      singleQubitMapping["physical_name"] = key;
      message["qubit_mapping"].push_back(singleQubitMapping);
    }

    ServerMessage yac = nlohmann::json::parse(circuitCode.code);
    yac["name"] = circuitCode.name;
    message["circuits"].push_back(yac);
    messages.push_back(message);
  }

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  // return the payload
  return std::make_tuple(iqmServerUrl + "circuits", headers, messages);
}

std::string IQMServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string IQMServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return "circuits" + postResponse["id"].get<std::string>() + "/counts";
}

std::string IQMServerHelper::constructGetJobPath(std::string &jobId) {
  return iqmServerUrl + "circuits/" + jobId + "/counts";
}

std::chrono::microseconds
IQMServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  return std::chrono::seconds(1); // jobs never take less than few seconds
};

bool IQMServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  cudaq::debug("getJobResponse: {}", getJobResponse.dump());

  auto jobStatus = getJobResponse["status"].get<std::string>();
  std::unordered_set<std::string> terminalStatuses = {"ready", "failed",
                                                      "aborted"};
  return terminalStatuses.find(jobStatus) != terminalStatuses.end();
}

cudaq::sample_result
IQMServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobID) {
  cudaq::info("postJobResponse: {}", postJobResponse.dump());

  // check if the job succeeded
  auto jobStatus = postJobResponse["status"].get<std::string>();
  if (jobStatus != "ready") {
    auto jobMessage = postJobResponse["message"].get<std::string>();
    throw std::runtime_error("Job status: " + jobStatus +
                             ", reason: " + jobMessage);
  }

  auto counts_batch = postJobResponse["counts_batch"];
  if (counts_batch.is_null()) {
    throw std::runtime_error("No counts in the response");
  }

  // assume there is only one measurement and everything goes into the
  // GlobalRegisterName of `sample_results`
  std::vector<ExecutionResult> srs;

  for (auto &counts : counts_batch.get<std::vector<ServerMessage>>()) {
    srs.push_back(ExecutionResult(
        counts["counts"].get<std::unordered_map<std::string, std::size_t>>()));
  }

  sample_result sampleResult(srs);

  // The original sampleResult is ordered by qubit number (FIXME: VERIFY THIS)
  // Now reorder according to reorderIdx[]. This sorts the global bitstring in
  // original user qubit allocation order.
  auto thisJobReorderIdxIt = reorderIdx.find(jobID);
  if (thisJobReorderIdxIt != reorderIdx.end()) {
    auto &thisJobReorderIdx = thisJobReorderIdxIt->second;
    if (!thisJobReorderIdx.empty())
      sampleResult.reorder(thisJobReorderIdx);
  }

  return sampleResult;
}

std::map<std::string, std::string>
IQMServerHelper::generateRequestHeader() const {
  std::map<std::string, std::string> headers{
      {"Content-Type", "application/json"},
      {"Connection", "keep-alive"},
      {"User-Agent", "cudaq/IQMServerHelper"},
      {"Accept", "*/*"}};

  // Prefer the authorization token set in the environment variable.
  if (authToken.has_value()) {
    headers["Authorization"] = "Bearer " + authToken.value();
  } else {
    // Fallback to authorization token from legacy JSON file.
    auto apiToken = readApiToken();
    if (apiToken.has_value()) {
      headers["Authorization"] = "Bearer " + apiToken.value();
    };
  }

  return headers;
}

void IQMServerHelper::updatePassPipeline(
    const std::filesystem::path &platformPath, std::string &passPipeline) {
  std::string pathToFile;

  auto iter = backendConfig.find("mapping_file");
  if (iter != backendConfig.end()) {
    // Use provided string as path+filename
    pathToFile = iter->second;
  } else {
    // Use the dynamic quantum architecture of the configured IQM server
    fetchQuantumArchitecture();
    pathToFile = writeQuantumArchitectureFile();
  }
  cudaq::info("Using quantum architecture file: {}", pathToFile);

  // Add leading and trailing single quotes to protect the filepath from
  // shell glob.
  pathToFile.insert(0, "'").append("'");

  passPipeline =
      std::regex_replace(passPipeline, std::regex("%QPU_ARCH%"), pathToFile);
}

/**
 * Fetch the quantum architecture from the configured URL and create a qubit
 * adjacency map.
 *
 * The qubit adjacency map contains only qubits which can be measured and can
 * be used in prx-gates as well as cz-gates. As qubits pairs for cz-gates
 * connect only a few qubits the information about neighbors is stored as sets
 * within a vector of all qubits to save memory.
 * @throws std::runtime_error thrown for any errors with the server.
 */
void IQMServerHelper::fetchQuantumArchitecture() {
  try {
    RestClient client;
    auto headers = generateRequestHeader();

    // From the Dynamic Quantum Architecture we need the list of qubits names,
    // the list of qubit pairs which can form cz-gates, the lists of qubits
    // which can do prx-gates and the list of qubits which support measurement.
    auto dynamicQuantumArchitecture = client.get(iqmServerUrl,
                                                 "calibration-sets/default/"
                                                 "dynamic-quantum-architecture",
                                                 headers);
    cudaq::debug("Dynamic QA={}", dynamicQuantumArchitecture.dump());

    auto &cz = dynamicQuantumArchitecture["gates"]["cz"];
    auto implementation = cz["default_implementation"];
    auto &cz_loci = cz["implementations"][implementation]["loci"];
    cudaq::debug("cz-gates ({}) loci={}", implementation, cz_loci.dump());

    auto &prx = dynamicQuantumArchitecture["gates"]["prx"];
    implementation = prx["default_implementation"];
    auto prx_loci = prx["implementations"][implementation]["loci"];
    cudaq::debug("prx-gates ({}) loci={}", implementation, prx_loci.dump());

    auto &measure = dynamicQuantumArchitecture["gates"]["measure"];
    implementation = measure["default_implementation"];
    auto &measure_loci = measure["implementations"][implementation]["loci"];
    cudaq::debug("measure ({}) loci={}", implementation, measure_loci.dump());

    // For each qubit set flags to indicate whether they can be used in cz,
    // prx or measurement operations. Then crop all qubits which are not
    // capable of all three operations and enumerate the remaining ones.

    for (auto qubit : dynamicQuantumArchitecture["qubits"]) {
      qubitNameMap[qubit] = 0; // initializing to zero meaning no capability
    }
    for (auto cz : cz_loci) {
      // each cz loci has 2 qubits - mark each qubit
      for (auto qubit : cz) { // cz is an array of strings
        qubitNameMap[qubit] |= 1 << 0;
      }
    }
    for (auto prx : prx_loci) {
      qubitNameMap[prx[0]] |= 1 << 1;
    }
    for (auto measure : measure_loci) {
      qubitNameMap[measure[0]] |= 1 << 2;
    }

    uint idx = 0; // enumeration counter
    for (auto qubit = qubitNameMap.begin(); qubit != qubitNameMap.end();) {
      if (qubit->second == 7) { // 7 = (1 << 0) | (1 << 1) | (1 << 2)
        qubit->second = idx++;  // replace flags with enumeration value
        qubit++;
      } else {
        qubit = qubitNameMap.erase(qubit);
      }
    }
    // From here on the qubitNameMap lists only qubits which can be used
    // for above listed operations. Starting with 0 each qubit in the list
    // is enumerated.

    // The number of qubits in this dynamic quantum architecture.
    uint qubitCount = qubitNameMap.size();
    cudaq::info("Server {} has {} calibrated qubits", iqmServerUrl, qubitCount);
    assert(idx == qubitCount);

#ifdef CUDAQ_DEBUG
    for (auto &[key, value] : qubitNameMap) {
      cudaq::debug("qubit mapping: {} = {}", key, value);
    }
#endif

    // Initialise the adjacency map with an empty set for each qubit
    qubitAdjacencyMap.reserve(qubitCount);
    for (uint i = 0; i < qubitCount; i++) {
      qubitAdjacencyMap.emplace_back();
    }

    // Iterate over all cz loci and add only those to the adjacency map
    // for which all qubits have passed the above tests.
    for (auto cz : cz_loci) {
      if (qubitNameMap.count(cz[0]) && qubitNameMap.count(cz[1])) {
        cudaq::debug("usable cz_loci {}", cz.dump());
        qubitAdjacencyMap[qubitNameMap[cz[0]]].insert(qubitNameMap[cz[1]]);
        qubitAdjacencyMap[qubitNameMap[cz[1]]].insert(qubitNameMap[cz[0]]);
      }
    } // for all cz loci
  } catch (const std::exception &e) {
    throw std::runtime_error("Unable to get quantum architecture from \"" +
                             iqmServerUrl + "\": " + std::string(e.what()));
  }
} // IQMServerHelper::fetchQuantumArchitecture()

/**
 * Write the content of the dynamic quantum architecture to file.
 *
 * The file is created in the system temporary file folder with an automatically
 * generated unique filename. The filename is returned by this function.
 *
 * @return String containing the generated filename.
 * @throws std::runtime_error thrown when file cannot be opened for writing.
 */
std::string IQMServerHelper::writeQuantumArchitectureFile(void) {
  uint qubitCount = qubitAdjacencyMap.size();

  // path to temporary file + unique filename
  quantumArchitectureFilePath =
      std::string(P_tmpdir) + "/qpu-architecture-XXXXXX";
  int fd = mkstemp(quantumArchitectureFilePath.data());
  if (fd < 0) {
    throw std::runtime_error("cannot write QPU architecture file " +
                             quantumArchitectureFilePath);
  }
  // open also as FILE which has better formatting with fprintf()
  FILE *file = fdopen(fd, "w");
  if (file == NULL) {
    throw std::runtime_error("cannot write QPU architecture file " +
                             quantumArchitectureFilePath);
  }

  // Header information
  fprintf(file, "# NOTE: automatically generated by " __FILE__ "\n");
  fprintf(file, "#       for server at URL: %s\n\n", iqmServerUrl.c_str());
  fprintf(file, "Number of nodes: %u\n", qubitCount);
  fprintf(file, "Number of edges: ?\n\n");

  // Write one line for each qubit listing the adjacent qubits.
  for (uint i = 0; i < qubitCount; i++) {
    bool first = true;

    std::string outputLine = std::to_string(i) + " --> {";
    for (uint node : qubitAdjacencyMap[i]) {
      if (first)
        first = false;
      else
        outputLine += ", ";
      outputLine += std::to_string(node);
    }
    outputLine += "}\n";

    fwrite(outputLine.c_str(), outputLine.length(), 1, file);
  }

  fclose(file);
  close(fd);

  return quantumArchitectureFilePath;
} // IQMServerHelper::writeQuantumArchitectureFile()

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IQMServerHelper, iqm)
