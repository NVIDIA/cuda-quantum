// QiboServerHelper.cpp
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/Support/Version.h"
#include "cudaq/utils/cudaq_utils.h"
#include <bitset>
#include <fstream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>
#include <unordered_set>

using json = nlohmann::json;

namespace cudaq {
std::string lowercaseArgument(std::string value) {
  std::transform(
    value.begin(), value.end(), value.begin(), [](unsigned char c){return std::tolower(c);}
  );
  return value;
}

bool booleanArgument(const std::string& string_argument) {
  return lowercaseArgument(string_argument) == "true";  // we should handle wrong string-boolean values
}

/// @brief The QiboServerHelper class extends the ServerHelper class
/// to handle interactions with the Qibo server for submitting and
/// retrieving quantum computation jobs.
class QiboServerHelper : public ServerHelper {
  static constexpr const char *DEFAULT_URL = "https://cloud.qibo.science";
  static constexpr const char *DEFAULT_VERSION = "0.2.2";

public:
  const std::string name() const override { return "qibo"; }

  /// @brief Example implementation of authentication headers.
  RestHeaders getHeaders() override {
    RestHeaders headers;
    headers["Content-Type"] = "application/json";

    // Add authentication headers if needed
    if (backendConfig.count("api_key"))
      headers["x-api-token"] = backendConfig["api_key"];
    headers["x-qibo-client-version"] = backendConfig["version"];

    return headers;
  }

  /// @brief Example implementation of backend initialization.
  void initialize(BackendConfig config) override {
    CUDAQ_INFO("Initializing Qibo Backend");
    backendConfig = config;

    if (!backendConfig.count("url"))
      backendConfig["url"] = DEFAULT_URL;
    if (!backendConfig.count("version"))
      backendConfig["version"] = DEFAULT_VERSION;
    if (!backendConfig.count("verbatim"))
      backendConfig["verbatim"] = "false";

    // Set shots if provided
    if (config.find("shots") != config.end())
      this->setShots(std::stoul(config["shots"]));
  }

  /// @brief Example implementation of simple job creation.
  ServerJobPayload createJob(std::vector<KernelExecution> &circuitCodes) override {
    ServerMessage job;

    job["circuit"] = circuitCodes[0].code;
    job["nshots"] = shots;
    job["device"] = backendConfig["device"];
    job["project"] = backendConfig["project"];
    job["verbatim"] = booleanArgument(backendConfig["verbatim"]);

    RestHeaders headers = getHeaders();
    std::string path = "/api/jobs";

    return std::make_tuple(backendConfig["url"] + path, headers,
                          std::vector<ServerMessage>{job});
  }

  /// @brief Example implementation of job ID tracking.
  std::string extractJobId(ServerMessage &postResponse) override {
    if (!postResponse.contains("pid"))
      return "";

    return postResponse.at("pid");
  }

  /// @brief Example implementation of job ID tracking.
  std::string constructGetJobPath(ServerMessage &postResponse) override {
    return extractJobId(postResponse);
  }

  /// @brief Example implementation of job ID tracking.
  std::string constructGetJobPath(std::string &jobId) override {
    return backendConfig["url"] + "/api/jobs/" + jobId;
  }

  /// @brief Example implementation of job status checking.
  bool jobIsDone(ServerMessage &getJobResponse) override {
    if (!getJobResponse.contains("status"))
      return false;

    std::string status = getJobResponse["status"];
    return status == "success" || status == "error";
  }

  /// @brief Example implementation of result processing.
  ///
  /// The raw results from quantum hardware often need post-processing (bit
  /// reordering, normalization, etc.) to match CUDA-Q's expectations.
  /// This is the place to do that.
  cudaq::sample_result processResults(ServerMessage &getJobResponse,
                                     std::string &jobId) override {
    CUDAQ_INFO("Processing results: {}", getJobResponse.dump());

    // Extract measurement results from the response
    auto samplesJson = getJobResponse["frequencies"];
    cudaq::CountsDictionary counts;

    for (auto &item : samplesJson.items()) {
      std::string bitstring = item.key();
      std::size_t count = item.value();
      counts[bitstring] = count;
    }

    // Create an ExecutionResult
    cudaq::ExecutionResult execResult{counts};

    // Return the sample_result
    return cudaq::sample_result{execResult};
  }

  /// @brief Example implementation of polling configuration.
  std::chrono::microseconds
  nextResultPollingInterval(ServerMessage &postResponse) override {
    return std::chrono::seconds(5);
  }
};

} // namespace cudaq

// Register the server helper in the CUDA-Q server helper factory
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::QiboServerHelper, qibo)