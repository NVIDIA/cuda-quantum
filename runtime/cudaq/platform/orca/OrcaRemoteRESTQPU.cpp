/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrcaRemoteRESTQPU.h"
#include "common/ServerHelper.h"
#include "orca_qpu.h"
#include "cudaq/runtime/logger/logger.h"
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
  serverHelper = cudaq::owning_ptr<ServerHelper>(
      registry::get<ServerHelper>(qpuName).release());
  serverHelper->initialize(backendConfig);

  // Give the server helper to the executor
  executor->setServerHelper(serverHelper.get());
}

cudaq::CompileTarget
cudaq::OrcaRemoteRESTQPU::getCompileTarget(const orca::sample_policy &) {
  CompileTarget target;
  target.overrideAOTCompilation = false;
  return target;
}

cudaq::detail::future
cudaq::OrcaRemoteRESTQPU::launchKernelCommon(const CompiledModule &module,
                                             KernelArgs args) {
  const auto &kernelName = module.getName();
  CUDAQ_INFO("OrcaRemoteRESTQPU: Launch kernel named '{}' remote QPU {}",
             kernelName, qpu_id);

  // TODO future iterations of this should support non-void return types.
  if (kernelName != orca::sample_policy::kernelName)
    throw std::runtime_error(kernelName + " is not supported on this target");

  auto packed = args.getPacked();
  if (!packed || packed->data.size() < sizeof(orca::TBIParameters))
    throw std::runtime_error("Orca launch requires packed TBI parameters.");

  auto params =
      *reinterpret_cast<const orca::TBIParameters *>(packed->data.data());
  return executor->execute(params, kernelName);
}

cudaq::orca::async_sample_policy::result_type
cudaq::OrcaRemoteRESTQPU::launchKernel(const orca::async_sample_policy &,
                                       const CompiledModule &module,
                                       KernelArgs args) {
  // Keep this asynchronous if requested
  return async_sample_result(launchKernelCommon(module, args));
}

cudaq::orca::sample_policy::result_type
cudaq::OrcaRemoteRESTQPU::launchKernel(const orca::sample_policy &,
                                       const CompiledModule &module,
                                       KernelArgs args) {
  // Otherwise make this synchronous
  auto result = launchKernelCommon(module, args).get();
  // TODO: support dynamic result types.
  return result;
}

void cudaq::OrcaRemoteRESTQPU::enqueue(cudaq::QuantumTask &task) {
  CUDAQ_INFO("OrcaRemoteRESTQPU: Enqueue Task on QPU {}", qpu_id);
  execution_queue->enqueue(task);
}

CUDAQ_REGISTER_TYPE(QPU, OrcaRemoteRESTQPU, orca)
