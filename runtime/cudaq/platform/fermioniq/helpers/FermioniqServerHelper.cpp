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
      "https://fermioniq-api-prod.azurewebsites.net";
  static constexpr const char *DEFAULT_API_KEY =
      "ZBwmQS4eR92BDnvz0B0QuSNBdLAydWKOlldLEGZ5sDxSAzFuvQB89A==";

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

  static constexpr const char *DEFAULT_REMOTE_CONFIG_ID =
      "8aa426bd-7a4e-4209-9f8a-746c40bc1816";

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

  /// @brief Return next results polling interval
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override;

private:
  /// @brief API Key for Fermioniq API
  std::string token;

  /// @brief user_id of logged in user
  std::string userId;

  std::vector<std::string> circuit_names;

  /// @brief exp time of token
  std::chrono::system_clock::time_point tokenExpTime;

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

// Initialize the Fermioniq server helper with a given backend configuration
void FermioniqServerHelper::initialize(BackendConfig config) {
  CUDAQ_INFO("Initializing Fermioniq Backend.");

  parseConfigForCommonParams(config);

  backendConfig[CFG_URL_KEY] =
      getEnvVar("FERMIONIQ_API_BASE_URL", DEFAULT_URL, false);
  backendConfig[CFG_API_KEY_KEY] =
      getEnvVar("FERMIONIQ_API_KEY", DEFAULT_API_KEY, false);

  backendConfig[CFG_ACCESS_TOKEN_ID_KEY] =
      getEnvVar("FERMIONIQ_ACCESS_TOKEN_ID", "", true);
  backendConfig[CFG_ACCESS_TOKEN_SECRET_KEY] =
      getEnvVar("FERMIONIQ_ACCESS_TOKEN_SECRET", "", true);

  backendConfig[CFG_USER_AGENT_KEY] =
      "cudaq/" + std::string(cudaq::getVersion());

  if (config.find("project_id") != config.end()) {
    backendConfig[CFG_PROJECT_ID_KEY] = config.at("project_id");
  }

  if (config.find("remote_config") != config.end()) {
    backendConfig[CFG_REMOTE_CONFIG_KEY] = config["remote_config"];
  } else {
    CUDAQ_INFO("Set default remote config {}", DEFAULT_REMOTE_CONFIG_ID);
    backendConfig[CFG_REMOTE_CONFIG_KEY] =
        std::string(DEFAULT_REMOTE_CONFIG_ID);
  }
  if (config.find("noise_model") != config.end()) {
    backendConfig[CFG_NOISE_MODEL_KEY] = config["noise_model"];
  }

  if (config.find("bond_dim") != config.end()) {
    backendConfig[CFG_BOND_DIM_KEY] = config.at("bond_dim");
  }

  refreshTokens(true);
}

// Implementation of the getValueOrDefault function
std::string FermioniqServerHelper::getValueOrDefault(
    const BackendConfig &config, const std::string &key,
    const std::string &defaultValue) const {
  return config.find(key) != config.end() ? config.at(key) : defaultValue;
}

std::vector<std::string> split_string(std::string str, std::string token) {
  std::vector<std::string> result;
  while (str.size()) {
    std::size_t index = str.find(token);
    if (index != std::string::npos) {
      result.push_back(str.substr(0, index));
      str = str.substr(index + token.size());
      if (str.size() == 0)
        result.push_back(str);
    } else {
      result.push_back(str);
      str = "";
    }
  }
  return result;
}

// Retrieve an environment variable
std::string FermioniqServerHelper::getEnvVar(const std::string &key,
                                             const std::string &defaultVal,
                                             const bool isRequired) const {
  // Get the environment variable
  const char *env_var = std::getenv(key.c_str());
  // If the variable is not set, either return the default or throw an exception
  if (env_var == nullptr) {
    if (isRequired)
      throw std::runtime_error(
          "The " + key + " environment variable is not set but is required.");
    else
      return defaultVal;
  }
  // Return the variable as a string
  return std::string(env_var);
}

// Helper function to get a value from a dictionary or return a default
template <typename K, typename V>
V getOrDefault(const std::unordered_map<K, V> &map, const K &key,
               const V &defaultValue) {
  auto it = map.find(key);
  return (it != map.end()) ? it->second : defaultValue;
}

// Check if a key exists in the backend configuration
bool FermioniqServerHelper::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

ServerJobPayload
FermioniqServerHelper::createJob(std::vector<KernelExecution> &circuitCodes) {
  CUDAQ_DBG("createJob");

  if (circuitCodes.size() != 1) {
    throw std::runtime_error("Fermioniq allows only one circuit codes.");
  }

  auto job = nlohmann::json::object();
  auto circuits = nlohmann::json::array({"__qir_base_compressed__"});
  auto configs = nlohmann::json::array();
  auto noise_models = nlohmann::json::array();

  std::vector<std::string> circuit_names;

  for (auto &circuitCode : circuitCodes) {
    CUDAQ_INFO("name: {}", circuitCode.name);

    circuit_names.push_back(circuitCode.name);

    CUDAQ_INFO("outputNames: {}", circuitCode.output_names.dump());

    // Construct the job message (for Fermioniq backend)
    circuits.push_back(circuitCode.code);

    auto config = nlohmann::json::object();
    config["n_shots"] = static_cast<int>(shots);
    if (keyExists(CFG_BOND_DIM_KEY)) {
      config["bond_dim"] = stoi(backendConfig.at(CFG_BOND_DIM_KEY));
    }

    if (circuitCode.user_data.contains("observable")) {
      config["observable"] = circuitCode.user_data["observable"];
    }

    configs.push_back(config);

    // To-DO: Add noise models
    noise_models.push_back(nullptr);
  }

  auto circuit_names_imploded = std::accumulate(
      circuit_names.begin(), circuit_names.end(), std::string(),
      [](const std::string &a, const std::string &b) -> std::string {
        return a + (a.length() > 0 ? "," : "") + b;
      });

  if (keyExists(CFG_REMOTE_CONFIG_KEY)) {
    job["remote_config"] = backendConfig.at(CFG_REMOTE_CONFIG_KEY);
  }
  job["circuit"] = circuits;
  job["config"] = configs;
  job["noise_model"] = noise_models;
  job["verbosity_level"] = 1;
  job["label"] = circuit_names_imploded;
  if (keyExists(CFG_PROJECT_ID_KEY)) {
    job["project_id"] = backendConfig.at(CFG_PROJECT_ID_KEY);
  }

  auto payload = nlohmann::json::array();
  payload.push_back(job);

  // Return a tuple containing the job path, headers, and the job message
  auto job_path = backendConfig.at(CFG_URL_KEY) + "/api/jobs";
  auto ret = std::make_tuple(job_path, getHeaders(), payload);
  return ret;
}

/// Refresh the api key and refresh-token
void FermioniqServerHelper::refreshTokens(bool force_refresh) {
  std::mutex m;
  std::lock_guard<std::mutex> l(m);
  RestClient client;

  if (!force_refresh) {
    auto now = std::chrono::system_clock::now();

    CUDAQ_DBG("now: {}, tokenExpTime: {}", now, tokenExpTime);

    auto timeLeft =
        std::chrono::duration_cast<std::chrono::minutes>(tokenExpTime - now);

    CUDAQ_DBG("timeleft minutes before token refresh: {}", timeLeft.count());

    if (timeLeft.count() <= 5) {
      force_refresh = true;
    }
  }

  if (!force_refresh) {
    return;
  }

  auto headers = getHeaders();
  nlohmann::json payload;
  payload["access_token_id"] = backendConfig.at(CFG_ACCESS_TOKEN_ID_KEY);
  payload["access_token_secret"] =
      backendConfig.at(CFG_ACCESS_TOKEN_SECRET_KEY);

  auto response_json = client.post(backendConfig.at(CFG_URL_KEY), "/api/login",
                                   payload, headers);
  token = response_json["jwt_token"].get<std::string>();
  userId = response_json["user_id"].get<std::string>();

  auto expDate = response_json["expiration_date"].get<std::string>();

  std::tm tm = {};
  // 2024-09-05T13:42:49.660841+00:00
  std::stringstream ss(expDate);
  ss >> std::get_time(&tm, "%Y-%m-%dT%H:%M:%S");

  tokenExpTime = std::chrono::system_clock::from_time_t(std::mktime(&tm));

  CUDAQ_DBG("exp time: {}", tokenExpTime);
}

bool FermioniqServerHelper::jobIsDone(ServerMessage &getJobResponse) {
#ifdef CUDAQ_DEBUG
  CUDAQ_DBG("check job status {}", getJobResponse.dump());
#endif

  refreshTokens(false);

  std::string status = getJobResponse.at("status");
  int status_code = getJobResponse.at("status_code");

  if (status == "finished") {
    CUDAQ_DBG("job is finished: {}", getJobResponse.dump());
    if (status_code == 0) {

      // label is where we store circuit names comma separated.
      std::string label = getJobResponse.at("job_label");
      auto splitted = split_string(label, ",");

      circuit_names = splitted;

      return true;
    }
    throw std::runtime_error("Job failed to execute. Status code = " +
                             std::to_string(status_code));
  } else {
    CUDAQ_INFO("job still running. status={}", status);
  }

  return false;
}

// From a server message, extract the job ID
std::string FermioniqServerHelper::extractJobId(ServerMessage &postResponse) {
  CUDAQ_DBG("extractJobId");

  return postResponse.at("id");
}

// Construct the path to get a job
std::string
FermioniqServerHelper::constructGetJobPath(ServerMessage &postResponse) {
  CUDAQ_DBG("constructGetJobPath");
  std::string id = postResponse.at("id");
  // todo: Extract job-id from postResponse

  auto ret = backendConfig.at(CFG_URL_KEY) + "/api/jobs/" + id;
  return ret;
}

// Overloaded version of constructGetJobPath for jobId input
std::string FermioniqServerHelper::constructGetJobPath(std::string &jobId) {
  CUDAQ_DBG("constructGetJobPath (jobId) from {}", jobId);

  auto ret = backendConfig.at(CFG_URL_KEY) + "/api/jobs/" + jobId;
  return ret;
}

// Process the results from a job
cudaq::sample_result
FermioniqServerHelper::processResults(ServerMessage &postJobResponse,
                                      std::string &jobID) {
  CUDAQ_DBG("processResults for job: {}", jobID);

  RestClient client;

  refreshTokens(false);

  auto headers = getHeaders();

  std::string path = "/api/jobs/" + jobID + "/results";

  auto response_json = client.get(backendConfig.at(CFG_URL_KEY), path, headers);

  CUDAQ_DBG("got job result: {}", response_json.dump());

  auto metadata = response_json.at("metadata");
  // CUDAQ_INFO("metadata: {}", metadata.dump());
  auto output = response_json.at("emulator_output");

  cudaq::sample_result sample_result;

  for (const auto &it : output.items()) {

    // "samples":{"00000":500,"11111":500}
    auto output = it.value().at("output");
    auto samples = output.at("samples");

    CountsDictionary sample_dict;
    for (const auto &[qubit_str, n_observed] : samples.items()) {
      sample_dict[qubit_str] = n_observed;
    }

    if (output.contains("expectation_values")) {
      auto exp_vals = output.at("expectation_values");

      double exp = exp_vals[0].at("expval").at("real");

      ExecutionResult exec_result(sample_dict, exp);
      sample_result = cudaq::sample_result(exec_result);
    } else {
      ExecutionResult exec_result(sample_dict);
      sample_result = cudaq::sample_result(exec_result);
    }

    break;
  }

  return sample_result;
}

RestHeaders FermioniqServerHelper::getHeaders() {

  RestHeaders headers;
  if (keyExists(CFG_API_KEY_KEY) && backendConfig.at(CFG_API_KEY_KEY) != "") {
    headers["x-functions-key"] = backendConfig.at(CFG_API_KEY_KEY);
  }

  if (!this->token.empty()) {
    headers["Authorization"] = token;
  }
  headers["Content-Type"] = "application/json";
  headers["User-Agent"] = backendConfig.at(CFG_USER_AGENT_KEY);

  // Return the headers
  return headers;
}

std::chrono::microseconds
FermioniqServerHelper::nextResultPollingInterval(ServerMessage &postResponse) {
  return std::chrono::seconds(POLLING_INTERVAL_IN_SECONDS);
};

} // namespace cudaq

// Register the Fermioniq server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::FermioniqServerHelper,
                    fermioniq)
