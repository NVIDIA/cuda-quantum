/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"

#include "nlohmann/json.hpp"

#include <fcntl.h>
#include <fstream>
#include <regex>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
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

  /// @brief Counter to limit the output of the status during polling
  uint statusOutputRateLimit = 0;

  /// @brief Number of total qubits on the addressed QPU
  uint qubitCountStaticArch = 0;

protected:
  /// @brief The base URL
  std::string iqmServerUrl = "http://localhost/";

  /// @brief The "id" or "alias" (name) of the quantum computer
  std::string iqmQC = "default";

  /// @brief Authorization token
  std::optional<std::string> authToken = std::nullopt;

  /// @brief The default cortex-cli tokens file path
  std::optional<std::string> tokensFilePath = std::nullopt;

  /// @brief Return the headers required for the REST calls
  RestHeaders generateRequestHeader() const;

  /// @brief Parse cortex-cli tokens JSON for the API access token
  std::optional<std::string> readApiToken() const {
    if (!tokensFilePath.has_value()) {
      CUDAQ_INFO(
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

  /// @brief Calibration-set ID from the dynamic quantum architecture
  std::string calibration_set_id = "";

  /// @brief The ID of the last job posted
  /// Cache here as the framework does not pass it to jobIdDone().
  std::string jobId;

  /// @brief Lookup table for translating the qubit names to index numbers
  std::map<std::string, uint, qubitOrder> qubitNameMap;

  /// @brief Adjacency map for each qubit
  std::vector<std::set<uint>> qubitAdjacencyMap;

  /// @brief full path+name of the file containing the quantum architecture
  std::string quantumArchitectureFilePath;

  /// @brief flag indicating that architecture file should be removed on exit
  bool cleanupQuantumArchitectureFilePath = true;

  /// @brief Fetch the quantum architecture from server
  void fetchQuantumArchitecture();

  /// @brief Reduce the topology to a single network
  void fixupTopology();

  /// @brief Write the dynamic quantum architecture file
  std::string writeQuantumArchitectureFile(void);

  /// @brief Read qubit mapping from quantum architecture file
  void readQuantumArchitectureFile(std::string filepath);

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

  /// @brief Return architecture-specific pipeline placeholder substitutions.
  std::map<std::string, std::string>
  getPipelineSubstitutions(const std::filesystem::path &platformPath) override;

  /// @brief Default destructor removes the dynamic quantum architecture file
  ~IQMServerHelper() {
    if ((cleanupQuantumArchitectureFilePath == true) &&
        (!quantumArchitectureFilePath.empty())) {
      if (unlink(quantumArchitectureFilePath.c_str()) != 0) {
        CUDAQ_INFO("Failed to delete {} with error: {}",
                   quantumArchitectureFilePath, std::string(strerror(errno)));
      }
    }
  }
};

void IQMServerHelper::initialize(BackendConfig config) {
  backendConfig = config;

  /* Configuration of URL and QC starts with default values. These can be
     overwritten in a first round with settings from the backend string.
     In a second round the values can be once more overwritten with settings
     from environment variables. This second round allows changing the target
     without recompilation or code changes. */

  // First apply values from the backend string if given.
  auto iter = backendConfig.find("url");
  if (iter != backendConfig.end()) {
    iqmServerUrl = iter->second;
  }
  iter = backendConfig.find("qc");
  if (iter != backendConfig.end()) {
    iqmQC = iter->second;
  }

  // Allow overriding IQM Server URL.
  auto envIqmServerUrl = getenv("IQM_SERVER_URL");
  if (envIqmServerUrl) {
    iqmServerUrl = std::string(envIqmServerUrl);
  }

  if (!iqmServerUrl.ends_with("/"))
    iqmServerUrl += "/";

  // For backward compatibility rewrite old style URLs.
  auto pos = iqmServerUrl.find("://cocos.");
  if (pos != std::string::npos) {
    iqmServerUrl.erase(pos + 3, 6); // skip the anchor and erase "cocos."
    pos = iqmServerUrl.find_first_of('/', pos + 3); // start of the path
    assert(pos != std::string::npos); // guaranteed by adding the slash above
    auto end = iqmServerUrl.find_first_of('/', pos + 1);
    if (end != std::string::npos) {
      // The old URL path starts with the QC alias.
      iqmQC = iqmServerUrl.substr(pos + 1, end - pos - 1);
      iqmServerUrl.erase(pos, end - pos);
    }
  }

  // Allow overriding the quantum computer selection.
  auto envIqmQc = getenv("IQM_QC"); // short hand for convenience
  if (envIqmQc) {
    iqmQC = std::string(envIqmQc);
  }
  envIqmQc = getenv("IQM_QUANTUM_COMPUTER"); // highest precedence
  if (envIqmQc) {
    iqmQC = std::string(envIqmQc);
  }

  CUDAQ_DBG("iqmServerUrl = {}", iqmServerUrl);
  CUDAQ_DBG("iqmQc = {}", iqmQC);

  auto token = getenv("IQM_TOKEN");
  if (token) {
    authToken = std::string(token);
    CUDAQ_DBG("Using authorization token from environment variable");
  } else {
    // Set alternative iqmclient-cli tokens file path if provided via env var
    auto envTokenFilePath = getenv("IQM_TOKENS_FILE");
    auto defaultTokensFilePath =
        std::string(getenv("HOME")) + "/.cache/iqm-client-cli/tokens.json";
    if (envTokenFilePath) {
      tokensFilePath = std::string(envTokenFilePath);
    } else if (cudaq::fileExists(defaultTokensFilePath)) {
      tokensFilePath = defaultTokensFilePath;
      CUDAQ_DBG("Setting default path for tokens file");
    }
    CUDAQ_DBG("tokensFilePath = {}", tokensFilePath.value_or("not set"));
  }

  token = getenv("IQM_SAVE_QPU_QA");
  if (token) {
    quantumArchitectureFilePath = std::string(token);
    cleanupQuantumArchitectureFilePath = false;
  }

  // Parse common config entries (e.g. `reorderIdx.<task_id>` populated by
  // Executor::execute) so that processResults() can map sampled bitstrings
  // back to the user's original qubit allocation order after the qubit
  // mapping pass has permuted them. Without this call the reorder map stays
  // empty and the bitstrings remain in physical-qubit order, which leads to
  // wrong bit positions when the mapping pass picks a non-identity
  // placement (see GitHub issue #4621).
  parseConfigForCommonParams(config);
}

ServerJobPayload
IQMServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  std::vector<ServerMessage> messages;

  // cuda-quantum expects every circuit to be a separate job,
  // so we cannot use the batch mode
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message = ServerMessage::object();
    message["circuits"] = ServerMessage::array();
    message["shots"] = shots;

    // Only add mapping if qubits were erased from the quantum architecture.
    if (qubitNameMap.size() != qubitCountStaticArch) {
      // Apply the mapping derived from the dynamic quantum architecture.
      message["qubit_mapping"] = ServerMessage::array();
      for (auto &[key, value] : qubitNameMap) {
        nlohmann::json singleQubitMapping;
        singleQubitMapping["logical_name"] = "QB" + std::to_string(value + 1);
        singleQubitMapping["physical_name"] = key;
        message["qubit_mapping"].push_back(singleQubitMapping);
      }
    }

    ServerMessage yac = nlohmann::json::parse(circuitCode.code);
    yac["name"] = circuitCode.name;
    message["circuits"].push_back(yac);
    messages.push_back(message);
  }

  // Get the headers
  RestHeaders headers = generateRequestHeader();

  // return the payload
  return std::make_tuple(iqmServerUrl + "api/v1/jobs/" + iqmQC + "/circuit",
                         headers, messages);
}

std::string IQMServerHelper::extractJobId(ServerMessage &postResponse) {
  return postResponse["id"].get<std::string>();
}

std::string IQMServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return "api/v1/jobs/" + postResponse["id"].get<std::string>();
}

std::string IQMServerHelper::constructGetJobPath(std::string &jobId) {
  this->jobId = jobId;
  return iqmServerUrl + "api/v1/jobs/" + jobId;
}

std::chrono::microseconds
IQMServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  uint delay = 1; // in seconds

  if (postResponse.contains("queue_position")) {
    uint pos = postResponse["queue_position"].get<uint>();
    CUDAQ_INFO("Queue position: {}", pos);
    while (pos > 3 && delay < 60) {
      pos--;
      delay += 5;
    }
  }

  if (delay > 30) {
    CUDAQ_INFO("Polling again in {} sec", delay);
  }
  return std::chrono::seconds(delay);
};

bool IQMServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  std::string jobStatus = getJobResponse["status"].get<std::string>();

  if (jobStatus != "waiting") {
    statusOutputRateLimit = 0;
    CUDAQ_INFO("Job Status: {}", jobStatus);
  } else {
    if (statusOutputRateLimit == 0) {
      statusOutputRateLimit = 10;
      CUDAQ_INFO("Job Status: {}", jobStatus);
    }
    statusOutputRateLimit--;
  }

  std::unordered_set<std::string> terminalStatuses = {"completed", "failed",
                                                      "cancelled"};
  bool done = terminalStatuses.find(jobStatus) != terminalStatuses.end();

  if (done) {
    // if the job failed exit with an exception
    if (jobStatus != "completed") {
      CUDAQ_INFO("getJobResponse: {}", getJobResponse.dump());
      std::string jobMessage = "unknown";
      try {
        jobMessage = getJobResponse["errors"][0]["message"].get<std::string>();
      } catch (const std::exception &e) {
        try {
          if (!getJobResponse["messages"].empty()) {
            jobMessage =
                getJobResponse["messages"][0]["message"].get<std::string>();
          }
        } catch (const std::exception &e) {
          jobMessage = "failed to get reason";
        }
      }
      throw std::runtime_error("Job status: " + jobStatus +
                               ", reason: " + jobMessage);
    }

    RestClient client;
    auto headers = generateRequestHeader();

    // retrieve the counts artifact
    ServerMessage counts_batch;
    try {
      counts_batch = client.get(
          iqmServerUrl,
          "api/v1/jobs/" + jobId + "/artifacts/measurement_counts", headers);
      if (counts_batch.is_null() || counts_batch.empty() ||
          counts_batch.type() != nlohmann::json::value_t::array ||
          counts_batch[0].type() != nlohmann::json::value_t::object) {
        throw std::runtime_error("No counts in the response");
      }
    } catch (const std::exception &e) {
      throw std::runtime_error("Unable to get counts for job " + jobId + ": " +
                               std::string(e.what()));
    }

    // Walk over the measurements and build a map for looking up on which qubit
    // the measurement with a given key is done. This is then appended to the
    // artifacts.
    try {
      ServerMessage job_payload;

      job_payload = client.get(iqmServerUrl,
                               "api/v1/jobs/" + jobId + "/payload", headers);
      CUDAQ_DBG("got payload: {}", job_payload.dump());

      nlohmann::json mkey2qubit;
      auto instructions = job_payload["circuits"][0]["instructions"];
      if (instructions.is_null()) {
        throw std::runtime_error("No circuit or instructions found");
      }

      for (auto instruction : instructions) {
        if (instruction["name"] == "measure") {
          mkey2qubit[instruction["args"]["key"]] = instruction["qubits"][0];
        }
      }
      counts_batch[0]["mkeys2qubit"] = mkey2qubit;
    } catch (const std::exception &e) {
      throw std::runtime_error("Unable to find circuit of job " + jobId + ": " +
                               std::string(e.what()));
    }

    CUDAQ_INFO("Artifacts: {}", counts_batch.dump());

    // replace the status request response with the counts artifacts
    getJobResponse = counts_batch;
  } // if (done)

  return done;
}

cudaq::sample_result
IQMServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobID) {
  // assume there is only one measurement and everything goes into the
  // GlobalRegisterName of `sample_results`
  std::vector<ExecutionResult> srs;

  for (auto &result : postJobResponse.get<std::vector<ServerMessage>>()) {
    if (!result.contains("measurement_keys")) {
      throw std::runtime_error("measurement_keys field missing in results");
    }
    if (!result.contains("mkeys2qubit")) {
      throw std::runtime_error("mkeys2qubit field missing in results");
    }
    std::map<std::string, std::string> mkeyLoci =
        result["mkeys2qubit"].get<std::map<std::string, std::string>>();
    std::map<std::string, std::size_t, qubitOrder> mKeys;
    std::vector<std::size_t> mOrder;
    std::size_t i = 0; // bit positions
    bool reorder = false;

    // The measurement_keys tell which qubits were measured. An ordered map
    // is used to sort the strings in numerical order and then the bitstrings
    // are ordered accordingly. As result the bitstrings are ordered according
    // to the physical qubit numbering.
    for (std::string key : result["measurement_keys"]) {
      // keys must not be empty and declared in the circuit
      if (key.empty() || mkeyLoci.count(key) == 0) {
        throw std::runtime_error("Undeclared measurement key received: " + key);
      }
      if (mKeys.count(mkeyLoci[key]) != 0) {
        // Must never happen as the "measurement_keys" in the circuit must be
        // unique. If the same name is used the transpiler adds a suffix.
        throw std::runtime_error("Duplicate measurement key in results: " +
                                 key);
      }
      mKeys[mkeyLoci[key]] = i++;
    }
    mOrder.reserve(mKeys.size());
    i = 0;
    for (auto [_, idx] : mKeys) {
      mOrder.push_back(idx);
      if (!reorder && idx != i++)
        reorder = true;
    }

    if (reorder) {
      std::unordered_map<std::string, std::size_t> cntDict;

      // get the bits into the order given by the measurement keys
      for (auto [bits, count] :
           result["counts"]
               .get<std::unordered_map<std::string, std::size_t>>()) {
        if (bits.size() != mOrder.size()) {
          throw std::runtime_error("Expected length " +
                                   std::to_string(mOrder.size()) +
                                   " for bitstring " + bits);
        }

        std::string oBits(bits);
        i = 0;
        for (auto idx : mOrder) {
          oBits[i++] = bits[idx];
        }

        cntDict[oBits] = count;
      }

      srs.push_back(ExecutionResult(cntDict));
    } else {
      srs.push_back(ExecutionResult(
          result["counts"]
              .get<std::unordered_map<std::string, std::size_t>>()));
    }
  }

  sample_result sampleResult(srs);

  // The original sampleResult is ordered by physical qubit number. Reorder
  // according to reorderIdx[] so the global bitstring is in the user's
  // original qubit allocation order.
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

/**
 * Put the quantum architecture file on the transpiler commandline.
 *
 * The quantum architecture file is normally a temporary file generated from
 * the dynamic quantum architecture which is retrieved from the configured
 * IQM server URL. But it can be overwritten by specifying the 'mapping_file'
 * parameter in the backend string, or even more flexible by setting the
 * environment variable 'IQM_QPU_QA' to the path+filename.
 */
std::map<std::string, std::string> IQMServerHelper::getPipelineSubstitutions(
    const std::filesystem::path &platformPath) {
  std::string pathToFile;

  // For normal operation the dynamic quantum architecture is retrieved from
  // the configured IQM server URL. This gives the connectivity of the
  // calibrated QPU.
  // For testing the file with the QPU quantum architecture can be given via
  // environment variable to allow testing different architectures without
  // recompilation. It can also be specified via the backend configuration.
  auto filename = getenv("IQM_QPU_QA");
  if (filename) {
    // Use provided string as path+filename
    pathToFile = std::string(filename);
    readQuantumArchitectureFile(pathToFile);
  } else {
    // Allow setting of quantum architecture file via the backend config
    auto iter = backendConfig.find("mapping_file");
    if (iter != backendConfig.end()) {
      // Use provided string as path+filename
      pathToFile = iter->second;
      readQuantumArchitectureFile(pathToFile);
    } else {
      // Use the dynamic quantum architecture of the configured IQM server.
      // Fallback to an empty substitution map and let the pipeline report the
      // problem if it is ever actually used.
      try {
        fetchQuantumArchitecture();
        fixupTopology();
        pathToFile = writeQuantumArchitectureFile();
      } catch (const std::exception &e) {
        CUDAQ_WARN("Leaving %QPU_ARCH% unresolved: {}. Set IQM_QPU_QA or pass "
                   "--mapping-file to supply it offline.",
                   e.what());
        return {};
      } catch (...) {
        CUDAQ_WARN("Leaving %QPU_ARCH% unresolved: Unable to get quantum "
                   "architecture for \"{}\" from \"{}\". Set IQM_QPU_QA or "
                   "pass --mapping-file to supply it offline.",
                   iqmQC, iqmServerUrl);
        return {};
      }
    }
  }
  CUDAQ_INFO("Using quantum architecture file: {}", pathToFile);

  // Add leading and trailing single quotes to protect the filepath from
  // shell glob.
  pathToFile.insert(0, "'").append("'");

  return {{"%QPU_ARCH%", pathToFile}};
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
    auto dynamicQuantumArchitecture =
        client.get(iqmServerUrl,
                   "api/v1/calibration-sets/" + iqmQC +
                       "/default/dynamic-quantum-architecture",
                   headers);

    CUDAQ_INFO("Dynamic QA={}", dynamicQuantumArchitecture.dump());

    calibration_set_id = dynamicQuantumArchitecture["calibration_set_id"];

    auto &cz = dynamicQuantumArchitecture["gates"]["cz"];

    auto &prx = dynamicQuantumArchitecture["gates"]["prx"];
    auto implementation = prx["default_implementation"];
    auto prx_loci = prx["implementations"][implementation]["loci"];

    auto &measure = dynamicQuantumArchitecture["gates"]["measure"];
    implementation = measure["default_implementation"];
    auto &measure_loci = measure["implementations"][implementation]["loci"];

    // For each qubit set flags to indicate whether they can be used in `cz`,
    // `prx` or `measurement` operations. Then crop all qubits which are not
    // capable of all three operations and enumerate the remaining ones.

    for (auto qubit : dynamicQuantumArchitecture["qubits"]) {
      qubitNameMap[qubit] = 0; // initializing to zero meaning no capability
    }
    qubitCountStaticArch = qubitNameMap.size();
    for (auto &cz_implementation : cz["implementations"]) {
      auto &cz_loci = cz_implementation["loci"];
      for (auto cz : cz_loci) {
        // each cz loci has 2 qubits - mark each qubit
        for (auto qubit : cz) { // cz is an array of strings
          qubitNameMap[qubit] |= (1 << 0);
        }
      }
    }
    for (auto prx : prx_loci) {
      qubitNameMap[prx[0]] |= (1 << 1);
    }
    for (auto measure : measure_loci) {
      qubitNameMap[measure[0]] |= (1 << 2);
    }

    uint idx = 0; // enumeration counter
    for (auto qubit = qubitNameMap.begin(); qubit != qubitNameMap.end();) {
      if (qubit->second == ((1 << 0) | (1 << 1) | (1 << 2))) {
        CUDAQ_DBG("mapping qubit: {} = {}", qubit->first, idx);
        qubit->second = idx++; // replace flags with enumeration value
        qubit++;
      } else {
        CUDAQ_DBG("SKIPPING qubit {} which lacks {}{}{}", qubit->first,
                  (qubit->second & (1 << 0) ? "" : "cz "),
                  (qubit->second & (1 << 1) ? "" : "prx "),
                  (qubit->second & (1 << 2) ? "" : "mx"));
        qubit = qubitNameMap.erase(qubit);
      }
    }
    // From here on the qubitNameMap lists only qubits which can be used
    // for all operations. Starting with 0 each qubit in the list
    // is enumerated.

    // The number of qubits in this dynamic quantum architecture.
    uint qubitCount = qubitNameMap.size();
    CUDAQ_INFO("Quantum computer \"{}\" at \"{}\" has {} calibrated qubits",
               iqmQC, iqmServerUrl, qubitCount);
    assert(idx == qubitCount);

    // Initialise the adjacency map with an empty set for each qubit
    qubitAdjacencyMap.clear();
    qubitAdjacencyMap.reserve(qubitCount);
    for (uint i = 0; i < qubitCount; i++) {
      qubitAdjacencyMap.emplace_back();
    }

    // Iterate over all cz loci of all implementations and add only those to
    // the adjacency map for which all qubits have passed the above tests.
    for (auto &cz_implementation : cz["implementations"]) {
      auto &cz_loci = cz_implementation["loci"];
      for (auto cz : cz_loci) {
        if (qubitNameMap.count(cz[0]) && qubitNameMap.count(cz[1])) {
          CUDAQ_DBG("usable cz_loci {}", cz.dump());
          qubitAdjacencyMap[qubitNameMap[cz[0]]].insert(qubitNameMap[cz[1]]);
          qubitAdjacencyMap[qubitNameMap[cz[1]]].insert(qubitNameMap[cz[0]]);
        }
      } // for all cz loci
    } // for all implementations

  } catch (const std::exception &e) {
    throw std::runtime_error("Unable to get quantum architecture for \"" +
                             iqmQC + "\" from \"" + iqmServerUrl +
                             "\": " + std::string(e.what()));
  } catch (...) {
    throw std::runtime_error("Unable to get quantum architecture for \"" +
                             iqmQC + "\" from \"" + iqmServerUrl + "\": ");
  }
} // IQMServerHelper::fetchQuantumArchitecture()

/**
 * Check for a split topology and if multiple networks exist remove all but one.
 *
 * The network remaining is either the one with the most nodes or if there is
 * a tie the one containing the qubit with the smallest index number.
 */
void IQMServerHelper::fixupTopology() {
  uint qubitCount = qubitAdjacencyMap.size();
  std::vector<uint> networkId;
  uint i, j;

  // Initially each qubit is a separate net and gets an own ID.
  networkId.reserve(qubitCount);
  for (i = 0; i < qubitCount; i++) {
    networkId.push_back(i);
  }

  // Iterate over the adjacency map and assign the same network ID to qubits
  // which are direct neighbours.
  uint touchedMaxQubit = 0;
  for (i = 0; i < qubitCount; i++) {
    for (auto j : qubitAdjacencyMap[i]) {
      // Only one direction of every connection needs to be checked.
      if (i < j) {
        // CUDAQ_DBG("qubit {} has nb {}", i, j);
        if (networkId[i] == networkId[j]) {
          // qubits already belong to the same network -> nothing to do
          continue;
        }

        // The lowest network id of both qubits will be used as id for
        // the merged network.
        uint newNetId = std::min(networkId[i], networkId[j]);

        // If the network id of the neighboring qubit is already modified all
        // touched qubits need to be checked for the network id to be replaced
        // and this then substituted with the id chosen for the merged network.
        if (networkId[j] != j) {
          uint prevNetId = std::max(networkId[i], networkId[j]);
          for (uint k = 0; k <= touchedMaxQubit; k++) {
            if (networkId[k] == prevNetId) {
              networkId[k] = newNetId;
            }
          }
        }

        // Set the same network id to both qubits.
        networkId[i] = networkId[j] = newNetId;

        if (touchedMaxQubit < j) {
          touchedMaxQubit = j;
        }
      }
    }
  }

#ifdef CUDAQ_DEBUG
  std::string listNetworkId = "";
  for (i = 0; i < qubitCount; i++) {
    listNetworkId += std::to_string(networkId[i]) + ", ";
  }
  CUDAQ_DBG("Network id's: {}", listNetworkId);
#endif

  /* Assumption is that there is a single contiguous network or not more than
     very few networks. This led to choosing a map for counting the qubits in
     each network. Drawback is that the map cannot be ordered by it's value
     but since we assume a few entries only iterating over them is fast. */

  // Count the number of qubits belonging to each network.
  std::map<uint, uint> nodeCnt;
  for (i = 0; i < qubitCount; i++) {
    nodeCnt[networkId[i]] += 1;
  }

  // Find the network with the largest number of qubits.
  uint maxCnt = 0, netId = 0;
  for (auto &[key, value] : nodeCnt) {
    CUDAQ_DBG("Network id {} has {} qubits", key, value);
    if (maxCnt < value) {
      maxCnt = value;
      netId = key;
    }
  }

  if (nodeCnt.size() > 1) {
    CUDAQ_INFO("Split topology detected! {} networks found.", nodeCnt.size());
    CUDAQ_DBG("Selected network id {} with {} qubits", netId, maxCnt);

    // Keep only the largest Network and drop all the other ones.

    for (i = qubitCount; i > 0; i--) {
      if (networkId[i - 1] != netId) {
        qubitAdjacencyMap.erase(qubitAdjacencyMap.begin() + (i - 1));
      }
    }

    uint idx = 0; // enumeration counter
    auto qubit = qubitNameMap.begin();
    for (i = 0; qubit != qubitNameMap.end(); i++) {
      if (networkId[i] == netId) {
        qubit->second = idx++;
        qubit++;
      } else {
        CUDAQ_DBG("dropping {}", qubit->first);
        qubit = qubitNameMap.erase(qubit);
      }
    }

    /* After erasing elements from the vectors with the results the sets with
       the indexes of the qubit neighbours need to be adjusted.
       The vector used above for counting the qubits is reused and prepared
       here as a lookup table for translating the initial qubit enumeration to
       the actual one. */
    for (i = j = 0; i < qubitCount; i++) {
      if (networkId[i] == netId) {
        // CUDAQ_DBG("qubit id {} -> {}", i, j);
        networkId[i] = j++;
      }
    }

    // After removing elements above get the new size here.
    qubitCount = qubitAdjacencyMap.size();
    CUDAQ_INFO("Reduced topology to largest network with {} qubit", qubitCount);

    // Translate the neighbour index numbers
    std::set<uint> neighbours;
    for (i = 0; i < qubitCount; i++) {
      // CUDAQ_DBG("qubit {}", i);
      neighbours.clear();
      for (uint nb : qubitAdjacencyMap[i]) {
        // CUDAQ_DBG(" nb {}", nb);
        neighbours.insert(networkId[nb]);
      }
      qubitAdjacencyMap[i] = neighbours;
    }
  }
}

/**
 * Write the content of the dynamic quantum architecture to file.
 *
 * If no filename is stored in 'quantumArchitectureFilePath' a unique filename
 * is generated automatically with a path in the system temporary file folder.
 * Next the file will be created and the dynamic quantum architecture written.
 * If the file already exists an error will be thrown. On success this function
 * returns the filename of the created file.
 *
 * @return String containing the filename of the created file.
 * @throws std::runtime_error thrown when file cannot be opened for writing
 *         or exists already.
 */
std::string IQMServerHelper::writeQuantumArchitectureFile(void) {
  uint qubitCount = qubitAdjacencyMap.size();
  int fd = -1;

  // open a file to write the dynamic quantum architecture to
  if (quantumArchitectureFilePath.empty()) {
    // if no filename is given a temporary file with unique name is generated
    quantumArchitectureFilePath =
        std::string(P_tmpdir) + "/qpu-architecture-XXXXXX";
    fd = mkstemp(quantumArchitectureFilePath.data());
  } else {
    fd = open(quantumArchitectureFilePath.data(), O_WRONLY | O_CREAT,
              S_IRUSR | S_IRGRP | S_IROTH);
  }
  if (fd < 0) {
    throw std::runtime_error("Cannot write QPU architecture file: \"" +
                             quantumArchitectureFilePath + "\" - " +
                             std::string(strerror(errno)));
  }
  if (ftruncate(fd, 0)) {
    throw std::runtime_error("Failed to truncate QPU architecture file: \"" +
                             quantumArchitectureFilePath + "\" - " +
                             std::string(strerror(errno)));
  }
  // open also as FILE which allows easier formatting with fprintf()
  FILE *file = fdopen(fd, "w");
  if (file == NULL) {
    throw std::runtime_error("Cannot write QPU architecture file: \"" +
                             quantumArchitectureFilePath + "\" - " +
                             std::string(strerror(errno)));
  }

  // Header
  fprintf(file,
          "# Automatically generated from calibration-set \"%s\" "
          "for quantum computer \"%s\" at IQM server URL: %s\n\n",
          calibration_set_id.c_str(), iqmQC.c_str(), iqmServerUrl.c_str());
  fprintf(file, "Number of nodes: %u\n", qubitCount);
  fprintf(file, "Number of edges: ?\n\n");

  std::string outputLine;

  // Write one line for each qubit listing the adjacent qubits.
  for (uint i = 0; i < qubitCount; i++) {
    bool first = true;

    outputLine = std::to_string(i) + " --> {";
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

  if (qubitNameMap.size() != qubitCountStaticArch) {
    fprintf(file, "\n# Mapping to physical qubit tags for IQM backend\n");

    outputLine = "# IQM qubit map:";
    for (auto &[key, value] : qubitNameMap) {
      outputLine += " \"" + key + "\"";
    }

    fwrite(outputLine.c_str(), outputLine.length(), 1, file);
  }

  fclose(file);
  close(fd);

  return quantumArchitectureFilePath;
} // IQMServerHelper::writeQuantumArchitectureFile()

/**
 * Read a qubit mapping list from a dynamic quantum architecture to file.
 *
 * Reads a qubit mapping list if such is included in the specified dynamic
 * quantum architecture file. The qubit mapping list is an IQM specific
 * extension of the quantum architecture file and located inside a comment.
 * The qubit mapping list is a sequence of strings in which each string is
 * enclosed in double quotes. It is loaded in order into the map used for
 * translating logical qubit numbers into physical qubit tags.
 *
 * @throws std::runtime_error thrown when file cannot be opened for reading.
 */
void IQMServerHelper::readQuantumArchitectureFile(std::string filepath) {
  std::fstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    throw std::runtime_error("Cannot read QPU architecture file: \"" +
                             filepath + "\" - " + std::string(strerror(errno)));
  }

  qubitNameMap.clear();

  while (std::getline(file, line)) {
    if (line.starts_with("# IQM qubit map:")) {
      CUDAQ_DBG("Loading qubit mapping from quantum architecture file");

      uint idx = 0; // enumeration counter for logical qubits
      size_t start = line.find_first_of(':'), end = start;

      // Parse for tags enclosed in a pair of double quotes.
      while (start != std::string::npos && end != std::string::npos) {
        start = line.find_first_of('"', end + 1);
        if (start != std::string::npos) {
          end = line.find_first_of('"', start + 1);
          if (end != std::string::npos) {
            qubitNameMap[line.substr(start + 1, end - start - 1)] = idx++;
          }
        }
      }
      break; // Process only the first occurrence of a mapping line.
    }
  }

  file.close();
}

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::IQMServerHelper, iqm)
