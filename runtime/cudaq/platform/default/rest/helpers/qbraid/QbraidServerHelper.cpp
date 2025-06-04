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

class QbraidServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://api.qbraid.com/api";
  static constexpr const char *DEFAULT_DEVICE = "ionq_simulator";
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
    backendConfig["job_path"] = backendConfig["url"] + "/quantum-jobs";
    backendConfig["results_path"] = backendConfig["url"] + "/quantum-jobs/result/";

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
      job["qbraidDeviceId"] = backendConfig.at("device_id");
      job["openQasm"] = circuitCode.code;
      job["shots"] = std::stoi(backendConfig.at("shots"));

      if (!circuitCode.name.empty()) {
        nlohmann::json tags;
        tags["name"] = circuitCode.name;
        job["tags"] = tags;
      }

      jobs.push_back(job);
    }

    return std::make_tuple(backendConfig.at("job_path"), getHeaders(), jobs);
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    if (!postResponse.contains("qbraidJobId")) {
      throw std::runtime_error("ServerMessage doesn't contain 'qbraidJobId' key.");
    }
    return postResponse.at("qbraidJobId");
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    if (!postResponse.contains("qbraidJobId")) {
      throw std::runtime_error("ServerMessage doesn't contain 'qbraidJobId' key.");
    }

    return backendConfig.at("job_path") + "?qbraidJobId=" + postResponse.at("qbraidJobId").get<std::string>();
  }

  std::string constructGetJobPath(std::string &jobId) override {
    return backendConfig.at("job_path") + "?qbraidJobId=" + jobId;
  }

  std::string constructGetResultsPath(const std::string &jobId) {
    return backendConfig.at("results_path") + jobId;
  }

  bool jobIsDone(ServerMessage &getJobResponse) override {
    std::string status;

    if (getJobResponse.contains("jobsArray") && !getJobResponse["jobsArray"].empty()) {
      status = getJobResponse["jobsArray"][0]["status"].get<std::string>();
      cudaq::info("Job status from jobs endpoint: {}", status);
    } else if (getJobResponse.contains("status")) {
      status = getJobResponse["status"].get<std::string>();
      cudaq::info("Job status from direct response: {}", status);
    } else if (getJobResponse.contains("data") && getJobResponse["data"].contains("status")) {
      status = getJobResponse["data"]["status"].get<std::string>();
      cudaq::info("Job status from data object: {}", status);
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

  // Sample results with results api - with retry logic
  cudaq::sample_result processResults(ServerMessage &getJobResponse, std::string &jobId) override {
    int maxRetries = 5;
    int waitTime = 2;
    float backoffFactor = 2.0;

    for (int attempt = 0; attempt < maxRetries; ++attempt) {
      try {
        auto resultsPath = constructGetResultsPath(jobId);
        auto headers = getHeaders();

        cudaq::info("Fetching results using direct endpoint (attempt {}/{}): {}", attempt + 1, maxRetries, resultsPath);
        RestClient client;
        auto resultJson = client.get("", resultsPath, headers, true);

        if (resultJson.contains("error") && !resultJson["error"].is_null()) {
          std::string errorMsg = resultJson["error"].is_string()
                                     ? resultJson["error"].get<std::string>()
                                     : resultJson["error"].dump();
          cudaq::info("Error from results endpoint: {}", errorMsg);

          if (attempt == maxRetries - 1) {
            throw std::runtime_error("Error retrieving results: " + errorMsg);
          }
        } else if (resultJson.contains("data") && resultJson["data"].contains("measurementCounts")) {
          cudaq::info("Processing results from direct endpoint");
          CountsDictionary counts;
          auto &measurements = resultJson["data"]["measurementCounts"];

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

        // If we get here, no valid data was found but also no error - retry
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0) ? waitTime : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("No valid results yet, retrying in {} seconds", sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }

      } catch (const std::exception &e) {
        cudaq::info("Exception when using direct results endpoint: {}", e.what());
        if (attempt < maxRetries - 1) {
          int sleepTime = (attempt == 0) ? waitTime : waitTime * std::pow(backoffFactor, attempt);
          cudaq::info("Retrying in {} seconds", sleepTime);
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        } else {
          cudaq::info("Falling back to original results processing method");
        }
      }
    }

    // Original result processing as fallback
    cudaq::info("Processing results from job response for job {}", jobId);
    if (getJobResponse.contains("jobsArray") && !getJobResponse["jobsArray"].empty()) {
      auto &job = getJobResponse["jobsArray"][0];

      if (job.contains("measurementCounts")) {
        CountsDictionary counts;
        auto &measurements = job["measurementCounts"];

        for (const auto &[bitstring, count] : measurements.items()) {
          counts[bitstring] = count.get<std::size_t>();
        }

        std::vector<ExecutionResult> execResults;
        execResults.emplace_back(ExecutionResult{counts});
        return cudaq::sample_result(execResults);
      }
    }

    // Last resort - check for direct measurementCounts in the response
    if (getJobResponse.contains("measurementCounts")) {
      CountsDictionary counts;
      auto &measurements = getJobResponse["measurementCounts"];

      for (const auto &[bitstring, count] : measurements.items()) {
        counts[bitstring] = count.get<std::size_t>();
      }

      std::vector<ExecutionResult> execResults;
      execResults.emplace_back(ExecutionResult{counts});
      return cudaq::sample_result(execResults);
    }

    throw std::runtime_error("No measurement counts found in any response format");
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
    headers["api-key"] = backendConfig.at("api_key");
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
