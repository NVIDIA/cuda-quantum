/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include <thread>
namespace cudaq {

/// @brief The OQCServerHelper class extends the ServerHelper class to handle
/// interactions with the OQC server for submitting and retrieving quantum
/// computation jobs.
class OQCServerHelper : public ServerHelper {
private:
  /// @brief RestClient used for HTTP requests.

  /// @brief Check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  /// @brief Output names indexed by jobID/taskID
  std::map<std::string, OutputNamesType> outputNames;

  /// @brief Create n requested tasks placeholders returning uuids for each
  std::vector<std::string> createNTasks(int n);

  /// @brief make a compiler config json string parameterising with number of
  /// shots
  std::string makeConfig(int shots);

public:
  RestClient client;

  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "oqc"; }

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

  /// @brief Constructs the URL for retrieving the results of a job based on the
  /// server's response to a job submission.
  std::string constructGetResultsPath(ServerMessage &postResponse);

  /// @brief Constructs the URL for retrieving the results of a job based on a
  /// job ID.
  std::string constructGetResultsPath(std::string &jobId);

  /// @brief Retrieves the results of a job using the provided path.
  ServerMessage getResults(std::string &resultsGetPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(ServerMessage &getJobResponse) override;

  /// @brief Processes the server's response to a job retrieval request and
  /// maps the results back to sample results.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) override;
};

namespace {

auto make_env_functor(std::string key, std::string def = "") {
  return [key, def]() {
    const char *env_var = std::getenv(key.c_str());
    // If the variable is not set, throw an exception
    if (env_var == nullptr && def.empty()) {
      throw std::runtime_error(key + " environment variable is not set.");
    }
    // Return the variable as a string
    return env_var != nullptr ? std::string(env_var) : def;
  };
}

std::string get_from_config(BackendConfig config, const std::string &key,
                            const auto &envVar) {
  const auto iter = config.find(key);
  return iter != config.end() ? iter->second : envVar();
}

} // namespace

// Initialize the OQC server helper with a given backend configuration
void OQCServerHelper::initialize(BackendConfig config) {

  cudaq::info("Initializing OQC Backend.");
  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    cudaq::info("Emulation is enabled, ignore all oqc connection specific "
                "information.");
    return;
  }
  // Set the necessary configuration variables for the OQC API
  config["url"] = get_from_config(
      config, "url",
      make_env_functor("OQC_URL", "https://sandbox.qcaas.oqc.app"));
  config["version"] = "v0.3";
  config["user_agent"] = "cudaq/0.3.0";
  config["target"] = "Lucy";
  config["qubits"] = 8;
  config["email"] =
      get_from_config(config, "email", make_env_functor("OQC_EMAIL"));
  config["password"] = make_env_functor("OQC_PASSWORD")();
  // Construct the API job path
  config["job_path"] = "/tasks"; // config["url"] + "/tasks";

  // Parse the output_names.* (for each job) and place it in outputNames[]
  for (auto &[key, val] : config) {
    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToBaseProfileQIR.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      for (const auto &el : outputNamesJSON[0]) {
        auto result = el[0].get<std::size_t>();
        auto qubit = el[1][0].get<std::size_t>();
        auto registerName = el[1][1].get<std::string>();
        jobOutputNames[result] = {qubit, registerName};
      }

      this->outputNames[key] = jobOutputNames;
    }
  }
  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Check if a key exists in the backend configuration
bool OQCServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

std::vector<std::string> OQCServerHelper::createNTasks(int n) {
  RestHeaders headers = OQCServerHelper::getHeaders();
  nlohmann::json j;
  std::vector<std::string> output;
  for (int i = 0; i < n; ++i) {
    auto response = client.post(backendConfig.at("url"),
                                backendConfig.at("job_path"), j, headers);
    output.push_back(response[0]);
  }
  return output;
}

std::string OQCServerHelper::makeConfig(int shots) {
  return "{\"$type\": \"<class 'scc.compiler.config.CompilerConfig'>\", "
         "\"$data\": {\"repeats\": " +
         std::to_string(shots) +
         ", \"repetition_period\": null, \"results_format\": {\"$type\": "
         "\"<class 'scc.compiler.config.QuantumResultsFormat'>\", \"$data\": "
         "{\"format\": {\"$type\": \"<enum "
         "'scc.compiler.config.InlineResultsProcessing'>\", \"$value\": 1}, "
         "\"transforms\": {\"$type\": \"<enum "
         "'scc.compiler.config.ResultsFormatting'>\", \"$value\": 3}}}, "
         "\"metrics\": {\"$type\": \"<enum "
         "'scc.compiler.config.MetricsType'>\", \"$value\": 6}, "
         "\"active_calibrations\": [], \"optimizations\": {\"$type\": \"<class "
         "'scc.compiler.config.Tket'>\", \"$data\": {\"tket_optimizations\": "
         "{\"$type\": \"<enum 'scc.compiler.config.TketOptimizations'>\", "
         "\"$value\": 30}}}}}";
}

// Create a job for the OQC quantum computer
ServerJobPayload
OQCServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("target") || !keyExists("qubits") || !keyExists("job_path"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");
  std::vector<ServerMessage> jobs(circuitCodes.size());
  std::vector<std::string> task_ids =
      OQCServerHelper::createNTasks(static_cast<int>(circuitCodes.size()));

  for (size_t i = 0; i < circuitCodes.size(); ++i) {
    nlohmann::json j;
    j["tasks"] = std::vector<nlohmann::json>();
    // Construct the job message
    nlohmann::json job;
    job["task_id"] = task_ids[i];
    job["config"] = makeConfig(static_cast<int>(shots));
    job["program"] = circuitCodes[i].code;
    j["tasks"].push_back(job);
    jobs[i] = j;
  }

  // Return a tuple containing the job path, headers, and the job message
  return std::make_tuple(backendConfig.at("url") +
                             backendConfig.at("job_path") + "/submit",
                         getHeaders(), jobs);
}

// From a server message, extract the job ID
std::string OQCServerHelper::extractJobId(ServerMessage &postResponse) {

  if (!postResponse.contains("task_id"))
    return "";

  // Return the job ID from the response
  return postResponse.at("task_id");
}

// Construct the path to get a job
std::string OQCServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  return backendConfig.at("job_path") + "/" +
         postResponse.at("task_id").get<std::string>() + "/all_info";
}

// Overloaded version of constructGetJobPath for jobId input
std::string OQCServerHelper::constructGetJobPath(std::string &jobId) {
  if (!keyExists("job_path")) {
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");
  }

  // Return the job path
  auto res = backendConfig.at("url") + backendConfig.at("job_path") + "/" +
             jobId + "/all_info";
  return res;
}

// Construct the path to get the results of a job
std::string
OQCServerHelper::constructGetResultsPath(ServerMessage &postResponse) {
  // Return the results path
  return backendConfig.at("job_path") + "/" +
         postResponse.at("task_id").get<std::string>() + "/all_info";
}

// Overloaded version of constructGetResultsPath for jobId input
std::string OQCServerHelper::constructGetResultsPath(std::string &jobId) {
  if (!keyExists("job_path"))
    throw std::runtime_error("Key 'job_path' doesn't exist in backendConfig.");

  return backendConfig.at("job_path") + "/" + jobId + "/all_info";
}

// Get the results from a given path
ServerMessage OQCServerHelper::getResults(std::string &resultsGetPath) {
  RestHeaders headers = getHeaders();
  // Return the results from the client
  return client.get(resultsGetPath, "", headers);
}

// Check if a job is done
bool OQCServerHelper::jobIsDone(ServerMessage &getJobResponse) {
  // Check if the necessary keys exist in the response
  if (!getJobResponse.contains("results"))
    throw std::runtime_error("ServerMessage doesn't contain 'results' key.");

  // Return whether the job is completed
  return !getJobResponse.at("results").is_null() ||
         !getJobResponse.at("task_error").is_null();
}

// Process the results from a job
cudaq::sample_result
OQCServerHelper::processResults(ServerMessage &postJobResponse,
                                std::string &jobId) {

  if (postJobResponse["results"].is_null()) {
    throw std::runtime_error("OQC backend error message: " +
                             postJobResponse["task_error"].dump());
  }
  cudaq::info("postJobResponse is {}", postJobResponse.dump());
  const auto &jsonResults = postJobResponse.at("results");

  cudaq::sample_result sampleResult; // value to return

  // Try to determine between two results formats:
  //   {"results":{"r0_r1_r2_r3_r4":{"00000":1000}}} (hasResultNames = true )
  //   {"results":{"00":479,"11":521}}               (hasResultNames = false)
  bool hasResultNames = false;
  for (const auto &element : jsonResults.items()) {
    // element.value() is either something like
    // {"00000":1000}} or 479
    if (element.value().is_object()) {
      hasResultNames = true;
      break;
    }
  }

  if (outputNames.find("output_names." + jobId) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobId);

  auto &output_names = outputNames["output_names." + jobId];
  for (auto &[result, info] : output_names) {
    cudaq::info("Qubit {} Result {} Name {}", info.qubitNum, result,
                info.registerName);
  }

  CountsDictionary countsDict;
  if (hasResultNames) {
    // The following code only supports 1 object in the returned results because
    // there is only 1 CountsDictionary and 1 register name in use, so throw a
    // warning if that isn't true.
    if (jsonResults.size() != 1)
      cudaq::info("WARNING: unexpected jsonResults size ({}). Continuing to "
                  "parse anyway.",
                  jsonResults.size());

    // Note: `name` contains a concatenated list of measurement names as
    // specified in the sent QIR program, separated by underscores. They are
    // ordered by QIR result number.
    // Example jsonResults: {"r0_r1_r2_r3_r4":{"00000":1000,"11111":1000}}
    for (const auto &[name, counts] : jsonResults.items()) {
      for (auto &element : counts.items())
        countsDict[element.key()] = element.value();
      ExecutionResult executionResult{counts, GlobalRegisterName};
      sampleResult.append(executionResult);
    }
  } else {
    // Example jsonResults: {"00":479,"11":521}
    for (auto &element : jsonResults.items())
      countsDict[element.key()] = element.value();
    ExecutionResult executionResult{countsDict, GlobalRegisterName};
    sampleResult.append(executionResult);
  }

  // Note: the bitstring is sorted by the underlying QIR result number. The user
  // doesn't know anything about QIR result numbers that the compiler generates,
  // so we need to convert them into something the user can understand. We will
  // make registers for each result using output_names.
  for (auto &[result, info] : output_names) {
    sample_result singleBitResult = sampleResult.get_marginal({result});
    ExecutionResult executionResult{singleBitResult.to_map(),
                                    info.registerName};
    sampleResult.append(executionResult);
  }

  // It does no good to return the global register to the user in result order
  // because the user doesn't know what result numbers the compiler ended up
  // using. Re-order global register to make it alphabetical based on result
  // name like our other emulation results.

  // Get the indices `idx[]` such that newBitStrings(:) = oldBitStr(idx(:)),
  // where newBitStrings will contain bitstrings that are alphabetically sorted
  // based on the result names.
  std::vector<std::size_t> idx(output_names.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::vector<std::string> outputNames(output_names.size());
  int i = 0;
  for (auto &[result, info] : output_names)
    outputNames[i++] = info.registerName;
  // Sort idx by outputNames
  std::sort(idx.begin(), idx.end(),
            [&outputNames](std::size_t a, std::size_t b) {
              return outputNames[a] < outputNames[b];
            });

  // Now reorder the bitstrings according to idx[]
  sampleResult.reorder(idx, GlobalRegisterName);

  return sampleResult;
}

// Get the headers for the API requests
RestHeaders OQCServerHelper::getHeaders() {
  // Check if the necessary keys exist in the configuration
  if (!keyExists("email") || !keyExists("password"))
    throw std::runtime_error("Key doesn't exist in backendConfig.");

  // Construct the headers
  RestHeaders headers;
  headers["Content-Type"] = "application/json";

  nlohmann::json j;
  j["email"] = backendConfig.at("email");
  j["password"] = backendConfig.at("password");
  nlohmann::json response =
      client.post(backendConfig.at("url") + "/auth", "", j, headers,
                  /*enableLossgging=*/false);
  std::string key = response.at("access_token");
  backendConfig["access_token"] = key;

  headers["Authorization"] = "Bearer " + backendConfig["access_token"];

  // Return the headers
  return headers;
}

} // namespace cudaq

// Register the OQC server helper in the CUDAQ server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::OQCServerHelper, oqc)
