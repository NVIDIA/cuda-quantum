
/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/FmtCore.h"

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq.h"
#include "nvqpp_config.h"

#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include "orca_qpu.h"

#include "llvm/Support/Base64.h"

#include <fstream>
#include <iostream>
#include <netinet/in.h>
#include <regex>
#include <sys/socket.h>
#include <sys/types.h>

namespace cudaq::orca {
cudaq::sample_result sample(std::vector<double> &bs_angles,
                            std::vector<double> &ps_angles,
                            std::vector<std::size_t> &input_state,
                            std::vector<std::size_t> &loop_lengths,
                            int n_samples) {
  TBIParameters parameters{bs_angles, ps_angles, input_state, loop_lengths,
                           n_samples};
  cudaq::ExecutionContext context("sample", n_samples);
  auto &platform = get_platform();
  platform.set_exec_ctx(&context, 0);
  cudaq::altLaunchKernel("orca_launch", nullptr, &parameters,
                         sizeof(TBIParameters), 0);

  return context.result;
}
} // namespace cudaq::orca

namespace {

/// @brief The OrcaRemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service.
/// Moreover, this QPU handles launching kernels under the Execution Context
/// that includs sampling via synchronous client invocations.
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
  }

  OrcaRemoteRESTQPU(OrcaRemoteRESTQPU &&) = delete;

  /// @brief The destructor
  virtual ~OrcaRemoteRESTQPU() = default;

  /// Enqueue a quantum task on the asynchronous execution queue.
  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supportsConditionalFeedback() override { return false; }

  /// Provide the number of shots
  void setShots(int _nShots) override { nShots = _nShots; }

  /// Clear the number of shots
  void clearShots() override { nShots = std::nullopt; }

  /// @brief Return true if the current backend is remote
  virtual bool isRemote() override { return !emulate; }

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
  cudaq::ServerJobPayload createJob(cudaq::orca::TBIParameters params);

  /// @brief Given a completed job response, map back to the sample_result
  cudaq::sample_result processResults(cudaq::ServerMessage &postJobResponse);

  /// @brief Returns the name of the server helper.
  const std::string name() const { return "orca"; }

  /// @brief Returns the headers for the server requests.
  cudaq::RestHeaders getHeaders();

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize();

  /// @brief Launch the kernel. Handle all pertinent
  /// modifications for the execution context.
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
    for (std::size_t i = 1; i < split.size(); i += 2) {
      // No need to decode trivial true/false values
      if (split[i + 1].starts_with("base64_")) {
        split[i + 1].erase(0, 7); // erase "base64_"
        std::vector<char> decoded_vec;
        if (auto err = llvm::decodeBase64(split[i + 1], decoded_vec))
          throw std::runtime_error("DecodeBase64 error");
        std::string decodedStr(decoded_vec.data(), decoded_vec.size());
        cudaq::info("Decoded {} parameter from '{}' to '{}'", split[i],
                    split[i + 1], decodedStr);
        backendConfig.insert({split[i], decodedStr});
      } else {
        backendConfig.insert({split[i], split[i + 1]});
      }
    }
  }

  /// Once we know the backend, we should search for the config file
  /// from there we can get the URL/PORT and other inforation used in the
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

/// @brief Launch the kernel.
void OrcaRemoteRESTQPU::launchKernel(const std::string &kernelName,
                                     void (*kernelFunc)(void *), void *args,
                                     std::uint64_t voidStarSize,
                                     std::uint64_t resultOffset) {
  cudaq::info("launching ORCA remote rest kernel ({})", kernelName);

  // TODO future iterations of this should support non-void return types.
  if (!executionContext)
    throw std::runtime_error("Remote rest execution can only be performed "
                             "via cudaq::sample() or cudaq::observe().");

  cudaq::orca::TBIParameters params =
      *((struct cudaq::orca::TBIParameters *)args);
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
  auto response = client.post(jobPostPath, "", job, headers);

  cudaq::sample_result counts = processResults(response);

  // // return the results synchronously
  executionContext->result = counts;
}

// Initialize the ORCA server helper with a given backend configuration
void OrcaRemoteRESTQPU::initialize() {
  // Set the machine
  auto iter = backendConfig.find("machine");
  if (iter != backendConfig.end())
    machine = iter->second;

  // Set a base URL if provided
  iter = backendConfig.find("url");
  if (iter != backendConfig.end()) {
    baseUrl = iter->second;
  }
}

// Create a job for the ORCA QPU
cudaq::ServerJobPayload
OrcaRemoteRESTQPU::createJob(cudaq::orca::TBIParameters params) {
  std::vector<cudaq::ServerMessage> jobs;
  cudaq::ServerMessage job;

  // Construct the job message
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