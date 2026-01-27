/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrcaRemoteRESTQPU.h"
#include "common/Logger.h"
#include "llvm/Support/Base64.h"

using namespace cudaq;

/// @brief This setTargetBackend override is in charge of reading the
/// specific target backend configuration file.
void cudaq::OrcaRemoteRESTQPU::setTargetBackend(const std::string &backend) {
  CUDAQ_INFO("OrcaRemoteRESTQPU platform is targeting {} with qpu_id = {}.",
             backend, qpu_id);

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
        CUDAQ_INFO("Decoded {} parameter from '{}' to '{}'", split[i],
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
  serverHelper = registry::get<ServerHelper>(qpuName);
  serverHelper->initialize(backendConfig);

  // Give the server helper to the executor
  executor->setServerHelper(serverHelper.get());
}

KernelThunkResultType cudaq::OrcaRemoteRESTQPU::launchKernelCommon(
    const std::string &kernelName, KernelThunkType kernelFunc, void *args) {

  CUDAQ_INFO("OrcaRemoteRESTQPU: Launch kernel named '{}' remote QPU {}",
             kernelName, qpu_id);

  auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
  auto ctx = contexts[tid];

  // TODO future iterations of this should support non-void return types.
  if (!ctx)
    throw std::runtime_error("Remote rest execution can only be performed "
                             "via cudaq::sample() or cudaq::observe().");

  orca::TBIParameters params = *((struct orca::TBIParameters *)args);
  std::size_t shots = params.n_samples;

  ctx->shots = shots;

  details::future future;
  future = executor->execute(params, kernelName);

  // Keep this asynchronous if requested
  if (ctx->asyncExec) {
    ctx->futureResult = future;
    return {};
  }

  // Otherwise make this synchronous
  ctx->result = future.get();

  // TODO: support dynamic result types.
  return {};
}

void cudaq::OrcaRemoteRESTQPU::launchKernel(const std::string &,
                                            const std::vector<void *> &) {
  throw std::runtime_error("launch kernel on raw args not implemented");
}

CUDAQ_REGISTER_TYPE(QPU, OrcaRemoteRESTQPU, orca)
