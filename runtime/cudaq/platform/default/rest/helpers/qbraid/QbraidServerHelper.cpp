#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/utils/cudaq_utils.h"
#include <thread>

namespace cudaq {

class QbraidServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api-v2.qbraid.com/api/v1";
  static constexpr const char *DEFAULT_DEVICE = "ionq:ionq:sim:simulator";
  static constexpr int DEFAULT_QUBITS = 29;

public:
  const std::string name() const override { return "qbraid"; }

  void initialize(BackendConfig config) override {
    cudaq::info("Initializing Qbraid Backend.");

    backendConfig.clear();
    backendConfig["url"] = getValueOrDefault(config, "url", DEFAULT_URL);
    backendConfig["device_id"] = getValueOrDefault(config, "device_id", DEFAULT_DEVICE);
    backendConfig["user_agent"] = "cudaq/" + std::string(cudaq::getVersion());
    backendConfig["qubits"] = std::to_string(DEFAULT_QUBITS);

    backendConfig["api_key"] = getEnvVar("QBRAID_API_KEY", "", true);
    backendConfig["job_path"] = backendConfig["url"] + "/jobs";

    backendConfig["results_output_dir"] = getValueOrDefault(config, "results_output_dir", "./qbraid_results");
    backendConfig["results_file_prefix"] = getValueOrDefault(config, "results_file_prefix", "qbraid_job_");

    if (!config["shots"].empty()) {
      backendConfig["shots"] = config["shots"];
      this->setShots(std::stoul(config["shots"]));
    } else {
      backendConfig["shots"] = "1000";
      this->setShots(1000);
    }

    parseConfigForCommonParams(config);

    cudaq::info("Qbraid configuration initialized:");
    for (const auto &[key, value] : backendConfig) {
      cudaq::info("  {} = {}", key, value);
    }

    std::string resultsDir = backendConfig["results_output_dir"];
    std::filesystem::create_directories(resultsDir);
    cudaq::info("Created results directory: {}", resultsDir);
  }

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    if (backendConfig.find("job_path") == backendConfig.end()) {
      throw std::runtime_error("job_path not found in config. Was initialize() called?");
    }

    std::vector<ServerMessage> jobs;
    for (auto &circuitCode : circuitCodes) {
      ServerMessage job;
      job["deviceQrn"] = backendConfig.at("device_id");
      job["shots"] = std::stoi(backendConfig.at("shots"));

      // v2 API: program is a structured object with format and data
      nlohmann::json program;
      program["format"] = "qasm2";
      program["data"] = circuitCode.code;
      job["program"] = program;

      // v2 API: name is a top-level field (not nested under tags)
      if (!circuitCode.name.empty()) {
        job["name"] = circuitCode.name;
      }

      jobs.push_back(job);
    }

    return std::make_tuple(backendConfig.at("job_path"), getHeaders(), jobs);
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    // v2 API: jobQrn is nested under data envelope
    if (postResponse.contains("data") && postResponse["data"].contains("jobQrn")) {
      return postResponse["data"]["jobQrn"].get<std::string>();
    }
    throw std::runtime_error("ServerMessage doesn't contain 'data.jobQrn' key.");
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    // v2 API: use path parameter instead of query parameter
    if (postResponse.contains("data") && postResponse["data"].contains("jobQrn")) {
      return backendConfig.at("job_path") + "/" + postResponse["data"]["jobQrn"].get<std::string>();
    }
    throw std::runtime_error("ServerMessage doesn't contain 'data.jobQrn' key.");
  }

  std::string constructGetJobPath(std::string &jobId) override {
    // v2 API: /jobs/{jobQrn}
    return backendConfig.at("job_path") + "/" + jobId;
  }

  std::string constructGetResultsPath(const std::string &jobId) {
    // v2 API: /jobs/{jobQrn}/result
    return backendConfig.at("job_path") + "/" + jobId + "/result";
  }

  std::string constructGetProgramPath(const std::string &jobId) {
    // v2 API: /jobs/{jobQrn}/program
    return backendConfig.at("job_path") + "/" + jobId + "/program";
  }

  bool jobIsDone(ServerMessage &getJobResponse) override {
    std::string status;

    // v2 API: status is nested under data envelope
    if (getJobResponse.contains("data") && getJobResponse["data"].contains("status")) {
      status = getJobResponse["data"]["status"].get<std::string>();
      cudaq::info("Job status from v2 data envelope: {}", status);
    } else if (getJobResponse.contains("status")) {
      // Fallback: direct status field
      status = getJobResponse["status"].get<std::string>();
      cudaq::info("Job status from direct response: {}", status);
    } else {
      cudaq::info("Unexpected job response format: {}", getJobResponse.dump());
      throw std::runtime_error("Invalid job response format");
    }

    if (status == "FAILED" || status == "COMPLETED" || status == "CANCELLED") {
      saveResponseToFile(getJobResponse);
      return true;
    }

    return false;
  }

  // Fetch the original program from v2 endpoint
  std::string getJobProgram(const ServerMessage &response, const std::string &jobId) override {
    auto programPath = constructGetProgramPath(jobId);
    auto headers = getHeaders();

    cudaq::info("Fetching job program from v2 endpoint: {}", programPath);
    RestClient client;
    auto programJson = client.get("", programPath, headers, true);

    // v2 API: program content at data.data, format at data.format
    if (programJson.contains("data") && programJson["data"].contains("data")) {
      cudaq::info("Retrieved program (format: {})",
                  programJson["data"].value("format", "unknown"));
      return programJson["data"]["data"].get<std::string>();
    }

    throw std::runtime_error("Invalid program response format: " + programJson.dump());
  }

  // Fetch results from v2 results endpoint with retry logic
  cudaq::sample_result processResults(ServerMessage &getJobResponse, std::string &jobId) override {
    int maxRetries = 5;
    int waitTime = 2;
    float backoffFactor = 2.0;

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
      try {
        auto resultsPath = constructGetResultsPath(jobId);
        auto headers = getHeaders();

        cudaq::info("Fetching results from v2 endpoint (attempt {}/{}): {}", attempt + 1, maxRetries, resultsPath);
        RestClient client;
        auto resultJson = client.get("", resultsPath, headers, true);

        // v2 API: error indicated by success=false
        if (resultJson.contains("success") && resultJson["success"].is_boolean()
            && !resultJson["success"].get<bool>()) {
          std::string errorMsg = "Results not yet available";
          if (resultJson.contains("data") && resultJson["data"].contains("message")) {
            errorMsg = resultJson["data"]["message"].get<std::string>();
          }
          cudaq::info("Results endpoint returned success=false: {}", errorMsg);

          if (attempt == maxRetries - 1) {
            throw std::runtime_error("Error retrieving results: " + errorMsg);
          }
        }
        // v2 API: measurementCounts nested under data.resultData
        else if (resultJson.contains("data")
                 && resultJson["data"].contains("resultData")
                 && resultJson["data"]["resultData"].contains("measurementCounts")) {
          cudaq::info("Processing results from v2 endpoint");
          CountsDictionary counts;
          auto &measurements = resultJson["data"]["resultData"]["measurementCounts"];

          for (const auto &[bitstring, count] : measurements.items()) {
            counts[bitstring] =
                count.is_number()
                    ? static_cast<std::size_t>(count.get<double>())
                    : static_cast<std::size_t>(count);
          }

          std::vector<ExecutionResult> execResults;
          execResults.emplace_back(ExecutionResult{counts});
          return cudaq::sample_result(execResults);
        }

        // No valid data yet and no explicit error - retry
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0) ? waitTime : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("No valid results yet, retrying in {} seconds", sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }

      } catch (const std::exception &e) {
        cudaq::info("Exception when fetching results: {}", e.what());
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0) ? waitTime : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("Retrying in {} seconds", sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }
      }
    }

    throw std::runtime_error("Failed to retrieve measurement counts after " +
                             std::to_string(maxRetries) + " attempts");
  }

  /// @brief Override the polling interval method
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(1);
  }

private:
  void saveResponseToFile(const ServerMessage &response, const std::string &identifier = "") {
    try {
      std::string outputDir = backendConfig.at("results_output_dir");
      std::string filePrefix = backendConfig.at("results_file_prefix");

      // Create a unique filename using timestamp if no identifier is provided
      std::string filename;
      if (identifier.empty()) {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
        filename = outputDir + "/" + filePrefix + std::to_string(timestamp) + ".json";
      } else {
        filename = outputDir + "/" + filePrefix + identifier + ".json";
      }

      std::ofstream outputFile(filename);
      if (!outputFile.is_open()) {
        cudaq::info("Failed to open file for writing: {}", filename);
        return;
      }

      outputFile << response.dump(2);
      outputFile.close();

      cudaq::info("Response saved to file: {}", filename);
    } catch (const std::exception &e) {
      cudaq::info("Error saving response to file: {}", e.what());
    }
  }

  RestHeaders getHeaders() override {
    if (backendConfig.find("api_key") == backendConfig.end()) {
      throw std::runtime_error("API key not found in config. Was initialize() called?");
    }

    RestHeaders headers;
    headers["X-API-KEY"] = backendConfig.at("api_key");
    headers["Content-Type"] = "application/json";
    headers["User-Agent"] = backendConfig.at("user_agent");
    return headers;
  }

  std::string getEnvVar(const std::string &key, const std::string &defaultVal, const bool isRequired) const {
    const char *env_var = std::getenv(key.c_str());
    if (env_var == nullptr) {
      if (isRequired) {
        throw std::runtime_error(key + " environment variable is not set.");
      }

      return defaultVal;
    }
    return std::string(env_var);
  }

  std::string getValueOrDefault(const BackendConfig &config,
                                const std::string &key,
                                const std::string &defaultValue) const {
    return config.find(key) != config.end() ? config.at(key) : defaultValue;
  }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QbraidServerHelper, qbraid)
