/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include <map>
#include <regex>
#include <sstream>
#include <thread>

namespace cudaq {

const std::string Lucy = "lucy";
const std::string Toshiko = "toshiko";
const std::map<std::string, uint> Machines = {{Lucy, 8}, {Toshiko, 32}};

/// @brief The OQCServerHelper class extends the ServerHelper class to handle
/// interactions with the OQC server for submitting and retrieving quantum
/// computation jobs.
class OQCServerHelper : public ServerHelper {
private:
  /// @brief RestClient used for HTTP requests.

  /// @brief Check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  /// @brief Create n requested tasks placeholders returning uuids for each
  std::vector<std::string> createNTasks(int n);

  /// @brief Gets endpoint of a device
  std::tuple<std::string, std::string>
      get_qpu_endpoint(std::string, std::string, std::string);

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

  /// @brief Update `passPipeline` with architecture-specific pass options
  void updatePassPipeline(const std::filesystem::path &platformPath,
                          std::string &passPipeline) override;
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
                            const auto &missing_functor) {
  const auto iter = config.find(key);
  auto item = iter != config.end() ? iter->second : missing_functor();
  std::transform(item.begin(), item.end(), item.begin(),
                 [](auto c) { return std::tolower(c); });
  return item;
}

std::string get_from_config_no_low(BackendConfig config, const std::string &key,
                                   const auto &missing_functor) {
  /// This function does the same with get_from_config, but without lowering
  /// Auth tokens are case sensitive and shouldn't be lowered

  const auto iter = config.find(key);
  auto item = iter != config.end() ? iter->second : missing_functor();
  std::transform(item.begin(), item.end(), item.begin(),
                 [](auto c) { return c; });
  return item;
}

void check_machine_allowed(const std::string &machine) {
  if (Machines.find(machine) == Machines.end()) {
    std::string allowed;
    for (const auto &machine : Machines)
      allowed += machine.first + " ";
    std::stringstream stream;
    stream << "machine " << machine
           << " is not of known oqc-machine set: " << allowed;
    throw std::runtime_error(stream.str());
  }
}

} // namespace

// Initialize the OQC server helper with a given backend configuration
void OQCServerHelper::initialize(BackendConfig config) {

  CUDAQ_INFO("Initializing OQC Backend.");

  // Fetch machine info before checking emulate because we want to be able to
  // emulate specific machines.
  auto machine = get_from_config(config, "machine", []() { return Lucy; });
  check_machine_allowed(machine);
  config["machine"] = machine;
  const auto emulate_it = config.find("emulate");
  if (emulate_it != config.end() && emulate_it->second == "true") {
    CUDAQ_INFO("Emulation is enabled, ignore all oqc connection specific "
               "information.");
    backendConfig = std::move(config);
    return;
  }

  config["entry_url"] =
      get_from_config(config, "entry_url", make_env_functor("OQC_URL"));
  config["target"] =
      get_from_config(config, "device", make_env_functor("OQC_DEVICE"));
  config["auth_token"] = get_from_config_no_low(
      config, "auth_token", make_env_functor("OQC_AUTH_TOKEN"));
  auto [device_url, dev_id] = get_qpu_endpoint(
      config["entry_url"], config["target"], config["auth_token"]);
  // Set the necessary configuration variables for the OQC API

  config["url"] = device_url;

  config["version"] = "v0.3";
  config["user_agent"] = "cudaq/0.3.0";
  config["oqc_user_agent"] = "QCaaS Client 3.9.1";
  config["qubits"] = Machines.at(machine);

  // Construct the API job path
  config["job_path"] = std::string("/") + dev_id + "/tasks";
  parseConfigForCommonParams(config);

  // Move the passed config into the member variable backendConfig
  backendConfig = std::move(config);
}

// Check if a key exists in the backend configuration
bool OQCServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

std::vector<std::string> OQCServerHelper::createNTasks(int n) {
  /// This function returns the task ids as a vector of string.
  /// The same as "create_task_ids" in QCaaS client, client.py

  RestHeaders headers = OQCServerHelper::getHeaders();
  nlohmann::json job;
  int nTask = n;
  job["task_count"] = nTask;
  job["qpu_id"] = backendConfig.at("target"); // qpu:uk:2:d865b5a184
  job["tag"] = "";
  std::vector<std::string> output;

  auto response = client.post(backendConfig.at("url"),
                              backendConfig.at("job_path"), job, headers);
  return response;
}

std::tuple<std::string, std::string>
OQCServerHelper::get_qpu_endpoint(std::string server_url, std::string qpu_id,
                                  std::string auth_token) {
  RestHeaders headers;

  headers["Authentication-Token"] = auth_token;
  auto response = client.get(server_url, "/admin/qpu", headers, true);

  for (auto item : response["items"]) {
    if (item["id"] == qpu_id) {
      std::string device_url = item["url"];
      size_t pos = device_url.find('/', 8); // Start fter "https://"
      std::string qpu_server_url = device_url.substr(0, pos);
      std::string qpu_device_id = device_url.substr(pos + 1);
      return std::make_tuple(qpu_server_url, qpu_device_id);
    }
  }

  std::stringstream stream;
  stream << "No device:" + qpu_id + " is on " + server_url + "." << std::endl;
  throw std::runtime_error(stream.str());
}

std::string OQCServerHelper::makeConfig(int shots) {
  return "{\"$type\": \"<class 'qat.purr.compiler.config.CompilerConfig'>\", "
         "\"$data\": {\"repeats\": " +
         std::to_string(shots) +
         ", \"repetition_period\": null, \"results_format\": {\"$type\": "
         "\"<class 'qat.purr.compiler.config.QuantumResultsFormat'>\", "
         "\"$data\": "
         "{\"format\": {\"$type\": \"<enum "
         "'qat.purr.compiler.config.InlineResultsProcessing'>\", \"$value\": "
         "1}, "
         "\"transforms\": {\"$type\": \"<enum "
         "'qat.purr.compiler.config.ResultsFormatting'>\", \"$value\": 3}}}, "
         "\"metrics\": {\"$type\": \"<enum "
         "'qat.purr.compiler.config.MetricsType'>\", \"$value\": 6}, "
         "\"active_calibrations\": [], \"optimizations\": {\"$type\": \"<class "
         "'qat.purr.compiler.config.Tket'>\", \"$data\": "
         "{\"tket_optimizations\": "
         "{\"$type\": \"<enum 'qat.purr.compiler.config.TketOptimizations'>\", "
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
    job["qpu_id"] = backendConfig.at("target");
    job["tag"] = "";
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
  CUDAQ_INFO("postJobResponse is {}", postJobResponse.dump());
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

  if (outputNames.find(jobId) == outputNames.end())
    throw std::runtime_error("Could not find output names for job " + jobId);

  auto &output_names = outputNames[jobId];
  for (auto &[result, info] : output_names) {
    CUDAQ_INFO("Qubit {} Result {} Name {}", info.qubitNum, result,
               info.registerName);
  }

  CountsDictionary countsDict;
  if (hasResultNames) {
    // The following code only supports 1 object in the returned results because
    // there is only 1 CountsDictionary and 1 register name in use, so throw a
    // warning if that isn't true.
    if (jsonResults.size() != 1)
      CUDAQ_INFO("WARNING: unexpected jsonResults size ({}). Continuing to "
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

  // First reorder the global register by QIR qubit number.
  std::vector<std::size_t> qirQubitMeasurements;
  qirQubitMeasurements.reserve(output_names.size());
  for (auto &[result, info] : output_names)
    qirQubitMeasurements.push_back(info.qubitNum);

  std::vector<std::size_t> idx(output_names.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](std::size_t i1, std::size_t i2) {
    return qirQubitMeasurements[i1] < qirQubitMeasurements[i2];
  });
  CUDAQ_INFO("Reordering global result to map QIR result order to QIR qubit "
             "allocation order is {}",
             idx);
  sampleResult.reorder(idx);

  // Now reorder according to reorderIdx[]. This sorts the global bitstring in
  // original user qubit allocation order.
  auto thisJobReorderIdxIt = reorderIdx.find(jobId);
  if (thisJobReorderIdxIt != reorderIdx.end()) {
    auto &thisJobReorderIdx = thisJobReorderIdxIt->second;
    if (!thisJobReorderIdx.empty())
      sampleResult.reorder(thisJobReorderIdx);
  }

  // We will also make registers for each result using output_names.
  for (auto &[result, info] : output_names) {
    sample_result singleBitResult = sampleResult.get_marginal({result});
    ExecutionResult executionResult{singleBitResult.to_map(),
                                    info.registerName};
    sampleResult.append(executionResult);
  }

  return sampleResult;
}

// Get the headers for the API requests
RestHeaders OQCServerHelper::getHeaders() {
  // Construct the headers
  RestHeaders headers;

  headers["Authentication-Token"] = backendConfig["auth_token"];
  headers["User-agent"] = backendConfig["oqc_user_agent"];
  headers["Content-type"] = "application/json";
  // Return the headers
  return headers;
}

void OQCServerHelper::updatePassPipeline(
    const std::filesystem::path &platformPath, std::string &passPipeline) {
  auto machine =
      get_from_config(backendConfig, "machine", []() { return Lucy; });
  check_machine_allowed(machine);
  std::string pathToFile = platformPath / std::string("mapping/oqc") /
                           (machine + std::string(".txt"));
  passPipeline =
      std::regex_replace(passPipeline, std::regex("%QPU_ARCH%"), pathToFile);
}

} // namespace cudaq

// Register the OQC server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::OQCServerHelper, oqc)
