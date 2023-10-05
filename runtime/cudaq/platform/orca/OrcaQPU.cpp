
/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// #include "common/ExecutionContext.h"
// #include "common/FmtCore.h"

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq.h"
// #include "cudaq/platform/default/rest/Executor.h"
// #include "cudaq/platform.h"
#include "nvqpp_config.h"

#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
// #include "cudaq/utils/cudaq_utils.h"

#include <fstream>
#include <iostream>
#include <regex>
// #include <thread>

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

  /// @brief The base URL
  std::string baseUrl;
  /// @brief The machine we are targeting
  std::string machine = "PT-1";

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
    // executor = std::make_unique<cudaq::Executor>();
  }

  OrcaRemoteRESTQPU(OrcaRemoteRESTQPU &&) = delete;

  virtual ~OrcaRemoteRESTQPU() = default;

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
    // executor->setShots(static_cast<std::size_t>(_nShots));
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

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file.
  void setTargetBackend(const std::string &backend) override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  cudaq::ServerJobPayload createJob(TBIParameters params); // void *kernelArgs);

  /// @brief Retrieves the results of a job using the provided path.
  cudaq::ServerMessage getResults(std::string &resultsGetPath);

  /// @brief Checks if a job is done based on the server's response to a job
  /// retrieval request.
  bool jobIsDone(cudaq::ServerMessage &getJobResponse);

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(cudaq::ServerMessage &postJobResponse);

  /// @brief Returns the name of the server helper.
  const std::string name() const { return "orca"; }

  /// @brief Returns the headers for the server requests.
  cudaq::RestHeaders getHeaders();

  // /// @brief Set the number of shots to execute
  // void setShots(std::size_t s) { shots = s; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize();

  /// @brief Launch the kernel. Extract the Quake code and lower to
  /// the representation required by the targeted backend. Handle all pertinent
  /// modifications for the execution context as well as async or sync
  /// invocation.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override;
};

/// @brief This setTargetBackend override is in charge of reading the
/// specific target backend configuration file.
void OrcaRemoteRESTQPU::setTargetBackend(const std::string &backend) {
  cudaq::info("Remote REST platform is targeting {}.", backend);

  // First we see if the given backend has extra config params
  auto mutableBackend = backend;
  if (mutableBackend.find(";") != std::string::npos) {
    auto split = cudaq::split(mutableBackend, ';');
    mutableBackend = split[0];
    // Must be key-value pairs, therefore an even number of values here
    if ((split.size() - 1) % 2 != 0)
      throw std::runtime_error(
          "Backend config must be provided as key-value pairs: " +
          std::to_string(split.size()));

    // Add to the backend configuration map
    for (std::size_t i = 1; i < split.size(); i += 2)
      backendConfig.insert({split[i], split[i + 1]});
  }

  /// Once we know the backend, we should search for the config file
  /// from there we can get the URL/PORT and the required MLIR pass
  /// pipeline.
  std::string fileName = mutableBackend + std::string(".config");
  auto configFilePath = platformPath / fileName;
  cudaq::info("Config file path = {}", configFilePath.string());
  std::ifstream configFile(configFilePath.string());
  std::string configContents((std::istreambuf_iterator<char>(configFile)),
                             std::istreambuf_iterator<char>());

  // Set the qpu name
  qpuName = mutableBackend;
  initialize();
}

/// @brief Launch the kernel. Extract the Quake code and lower to
/// the representation required by the targeted backend. Handle all pertinent
/// modifications for the execution context as well as async or sync
/// invocation.
void OrcaRemoteRESTQPU::launchKernel(const std::string &kernelName,
                                     void (*kernelFunc)(void *), void *args,
                                     std::uint64_t voidStarSize,
                                     std::uint64_t resultOffset) {
  cudaq::info("launching ORCA remote rest kernel ({})", kernelName);

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
  auto job = jobs[0];
  cudaq::info("Job (name={}) created, posting to {}", kernelName, jobPostPath);

  // Post it, get the response
  auto response1 = client.post(jobPostPath, "", job, headers);

  cudaq::sample_result counts = processResults(response1);

  // // make this synchronous
  executionContext->result = counts;
}

// Initialize the ORCA server helper with a given backend configuration
void OrcaRemoteRESTQPU::initialize() {
  // Set the machine
  auto iter = backendConfig.find("machine");
  if (iter != backendConfig.end())
    machine = iter->second;

  // Set an alternate base URL if provided
  iter = backendConfig.find("url");
  if (iter != backendConfig.end()) {
    baseUrl = iter->second;
  }
}

// Check if a key exists in the backend configuration bool
bool OrcaRemoteRESTQPU::keyExists(const std::string &key) const {
  return backendConfig.find(key) != backendConfig.end();
}

// Create a job for the ORCA quantum computer
cudaq::ServerJobPayload OrcaRemoteRESTQPU::createJob(TBIParameters params) {
  std::vector<cudaq::ServerMessage> jobs;
  // Construct the job message
  cudaq::ServerMessage job;
  job["target"] = machine;

  job["bs_angles"] = params.bs_angles;
  job["ps_angles"] = params.ps_angles;
  job["input_state"] = params.input_state;
  job["loop_lengths"] = params.loop_lengths;
  job["n_samples"] = params.n_samples;

  jobs.push_back(job);

  // Return a tuple containing the job path, headers, and the job message
  auto ret = std::make_tuple(baseUrl, getHeaders(), jobs);
  return ret;
}

// Get the results from a given path
cudaq::ServerMessage
OrcaRemoteRESTQPU::getResults(std::string &resultsGetPath) {
  cudaq::RestHeaders headers = getHeaders();
  // Return the results from the client
  return client.get(resultsGetPath, "", headers);
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
OrcaRemoteRESTQPU::processResults(cudaq::ServerMessage &postJobResponse) {
  auto results = postJobResponse.at("results");

  cudaq::CountsDictionary counts;
  // Process the results
  for (const auto &key : results) {
    counts[key] += 1;
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