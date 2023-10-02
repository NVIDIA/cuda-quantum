
/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/platform.h"
#include "cudaq/platform/default/rest/Executor.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"

#include <fstream>
#include <iostream>
#include <regex>
#include <thread>

namespace {

struct TBIParameters {
  std::vector<double> bs_angles;
  std::vector<double> ps_angles;

  std::vector<std::size_t> input_state;
  std::vector<std::size_t> loop_lengths;

  int n_samples;
};

/// @brief The OrcaRemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA Quantum kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service. Specific
/// details about JSON payloads are abstracted via an abstract type called
/// ServerHelper, which is meant to be subtyped by each provided remote QPU
/// service. Moreover, this QPU handles launching kernels under a number of
/// Execution Contexts, including sampling and observation via synchronous or
/// asynchronous client invocations. This type should enable both QIR-based
/// backends as well as those that take OpenQASM2 as input.
class OrcaRemoteRESTQPU : public cudaq::QPU {
protected:
  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  // Pointer to the concrete Executor for this QPU
  std::unique_ptr<cudaq::Executor> executor;

  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Flag indicating whether we should emulate
  /// execution locally.
  bool emulate = false;

private:
  /// @brief RestClient used for HTTP requests.
  cudaq::RestClient client;

public:
  /// @brief The constructor
  OrcaRemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<cudaq::Executor>();
  }

  OrcaRemoteRESTQPU(OrcaRemoteRESTQPU &&) = delete;

  ~OrcaRemoteRESTQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  /// @return
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supportsConditionalFeedback() override { return false; }

  /// Provide the number of shots
  void setShots(int _nShots) override {
    nShots = _nShots;
    executor->setShots(static_cast<std::size_t>(_nShots));
  }

  /// Clear the number of shots
  void clearShots() override { nShots = std::nullopt; }
  virtual bool isRemote() override { return !emulate; }

  /// @brief Helper method to check if a key exists in the configuration.
  bool keyExists(const std::string &key) const;

  /// Store the execution context for launchKernel
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    if (!context)
      return;

    cudaq::info("Remote Rest QPU setting execution context to {}",
                context->name);

    // Execution context is valid
    executionContext = context;
  }

  /// Reset the execution context
  void resetExecutionContext() override {
    // do nothing here
    executionContext = nullptr;
  }

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  cudaq::ServerJobPayload createJob(TBIParameters params); // void *kernelArgs);

  /// @brief Retrieves the results of a job using the provided path.
  cudaq::ServerMessage getResults(std::string &resultsGetPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(cudaq::ServerMessage &getJobResponse);

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(cudaq::ServerMessage &postJobResponse,
                                      std::string &jobId);

  /// @brief Returns the name of the server helper.
  const std::string name() const { return "orca"; }

  /// @brief Returns the headers for the server requests.
  cudaq::RestHeaders getHeaders();

  // /// @brief Set the number of shots to execute
  // void setShots(std::size_t s) { shots = s; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(cudaq::BackendConfig config);

  /// @brief Launch the kernel. Extract the Quake code and lower to
  /// the representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as async or sync
  /// invocation.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("launching remote rest kernel ({})", kernelName);

    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error("Remote rest execution can only be performed "
                               "via cudaq::sample() or cudaq::observe().");

    TBIParameters params = *((struct TBIParameters *)args);
    std::size_t shots = params.n_samples;

    setShots(shots);
    executionContext->shots = shots;

    cudaq::info("Executor creating job to execute with the {} helper.", name());

    // Create the Job Payload, composed of job post path, headers,
    // and the job json messages themselves
    auto [jobPostPath, headers, jobs] = createJob(params);

    cudaq::info("Job (name={}) created, posting to {}", kernelName,
                jobPostPath);

    // Post it, get the response
    cudaq::ServerMessage response;
    response["id"] = "0";
    response["results_url"] = "https://xxx.xx";

    // auto response = client.post(jobPostPath, "", job, headers);
    cudaq::info("Job (name={}) posted, response was {}", kernelName,
                response.dump());

    cudaq::ServerMessage res = {
        {"id", "0"},
        {"jobs",
         {
             {{"status", "completed"}, {"results_url", "/v0.3/jobs/0/results"}},
         }}};
    std::string jobId = "0";
    cudaq::sample_result counts = processResults(res, jobId);

    // // make this synchronous
    executionContext->result = counts;
  }
};

// Initialize the ORCA server helper with a given backend configuration
void OrcaRemoteRESTQPU::initialize(cudaq::BackendConfig config) {
  cudaq::info("Initializing ORCA Backend.");
  // Move the passed config into the member variable backendConfig
  // Set the necessary configuration variables for the ORCA API
  backendConfig["url"] =
      config.find("url") != config.end() ? config["url"] : "https://xxx.xxx.xx";
  backendConfig["version"] = "v0.3";
  backendConfig["user_agent"] = "cudaq/0.3.0";
  backendConfig["target"] =
      config.find("qpu") != config.end() ? config["qpu"] : "simulator";
  backendConfig["qubits"] = 29;
  // Retrieve the noise model setting (if provided)
  if (config.find("noise") != config.end())
    backendConfig["noise_model"] = config["noise"];
  // Retrieve the API key from the environment variables
  // backendConfig["token"] = getEnvVar("ORCA_API_KEY");
  // Construct the API job path
  backendConfig["job_path"] =
      backendConfig["url"] + '/' + backendConfig["version"] + "/jobs";
}

// Check if a key exists in the backend configuration bool
bool OrcaRemoteRESTQPU::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

// Create a job for the ORCA quantum computer
cudaq::ServerJobPayload OrcaRemoteRESTQPU::createJob(TBIParameters params) {

  //   cudaq::ServerJobPayload OrcaRemoteRESTQPU::createJob(void *kernelArgs) {

  //   TBIParameters params = *((struct TBIParameters *)kernelArgs);

  // Construct the job message
  cudaq::ServerMessage job;
  job["target"] = "PT-Series";

  job["bs_angles"] = params.bs_angles;
  job["ps_angles"] = params.ps_angles;
  job["input_state"] = params.input_state;
  job["loop_lengths"] = params.loop_lengths;
  job["n_samples"] = params.n_samples;

  // std::cout << job << std::endl;

  // Return a tuple containing the job path, headers, and the job message
  auto ret = std::make_tuple("http://localhost:8080/sample", getHeaders(), job);
  return ret;
}

// Get the results from a given path
cudaq::ServerMessage
OrcaRemoteRESTQPU::getResults(std::string &resultsGetPath) {
  cudaq::RestHeaders headers = getHeaders();
  // Return the results from the client
  cudaq::ServerMessage resp;
  resp["02"] = 1000;
  resp["11"] = 5000;
  resp["20"] = 2000;
  return resp;
  // return client.get(resultsGetPath, "", headers);
}

// Check if a job is done
bool OrcaRemoteRESTQPU::jobIsDone(cudaq::ServerMessage &getJobResponse) {
  // Check if the necessary keys exist in the response
  if (!getJobResponse.contains("jobs"))
    throw std::runtime_error("ServerMessage doesn't contain 'jobs' key.");

  auto &jobs = getJobResponse.at("jobs");

  if (jobs.empty() || !jobs[0].contains("status"))
    throw std::runtime_error("cudaq::ServerMessage doesn't contain 'status' "
                             "key in the first job.");

  // Throw a runtime error if the job has failed
  if (jobs[0].at("status").get<std::string>() == "failed")
    throw std::runtime_error("The job failed upon submission. Check the job "
                             "submission in your ORCA "
                             "account for more information.");

  // Return whether the job is completed
  return jobs[0].at("status").get<std::string>() == "completed";
}

// Process the results from a job
cudaq::sample_result
OrcaRemoteRESTQPU::processResults(cudaq::ServerMessage &postJobResponse,
                                  std::string &jobId) {
  // Construct the path to get the results
  auto resultsGetPath = constructGetResultsPath(postJobResponse);
  // Get the results
  auto results = getResults(resultsGetPath);
  cudaq::CountsDictionary counts;

  // Process the results
  for (const auto &element : results.items()) {
    std::string key = element.key();
    double value = element.value().get<double>();
    std::size_t count = static_cast<std::size_t>(value);
    counts[key] = count;
  }

  // Create an execution result
  cudaq::ExecutionResult executionResult(counts);
  // Return a sample result
  auto ret = cudaq::sample_result(executionResult);
  return ret;
}

// Get the headers for the API requests
cudaq::RestHeaders OrcaRemoteRESTQPU::getHeaders() {
  // Construct the headers
  cudaq::RestHeaders headers;
  headers["Authorization"] = "apiKey ";
  headers["Content-Type"] = "application/json";
  // Return the headers
  return headers;
}

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, OrcaRemoteRESTQPU, orca)