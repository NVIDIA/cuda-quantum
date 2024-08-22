/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "OrcaRemoteRESTQPU.h"
#include "common/Future.h"
#include "common/Logger.h"
#include "common/Registry.h"
#include "llvm/Support/Base64.h"

namespace cudaq{
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
  /// from there we can get the URL/PORT and other information used in the
  /// pipeline.
  // Set the qpu name
  qpuName = mutableBackend;
  serverHelper = registry::get<OrcaServerHelper>(qpuName);
  serverHelper->initialize(backendConfig);

  // Give the server helper to the executor
  executor->setServerHelper(serverHelper.get());
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

  cudaq::details::future future;
  future = executor->execute(params);

  // Keep this asynchronous if requested
  if (executionContext->asyncExec) {
    executionContext->futureResult = future;
    return;
  }

  // Otherwise make this synchronous
  executionContext->result = future.get();

  // // Create the Job Payload, composed of job post path, headers,
  // // and the job json messages themselves
  // auto [jobPostPath, headers, jobs] = serverHelper->createJob(params);
  // auto job = jobs[0];
  // cudaq::info("Job (name={}) created, posting to {}", kernelName,
  // jobPostPath);

  // // Post it, get the response
  // auto response = client.post(jobPostPath, "", job, headers);
  // cudaq::info("Job (name={}) posted, response was {}", kernelName,
  //             response.dump());

  // cudaq::sample_result counts = serverHelper->processResults(response);

  // // return the results synchronously
  // executionContext->result = counts;
}

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::OrcaRemoteRESTQPU, orca)