/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

//===----------------------------------------------------------------------===//
//
// Simple interfaces for remote server-client execution
//
//===----------------------------------------------------------------------===//

#pragma once

#include "common/Registry.h"
#include "cudaq/remote_capabilities.h"
#include <optional>
#include <string_view>
#include <unordered_map>

namespace mlir {
class MLIRContext;
}
namespace cudaq {
class ExecutionContext;
class gradient;
class optimizer;
class SerializedCodeExecutionContext;

/// Base interface encapsulating a CUDA-Q runtime server capable of
/// running kernel IR code.
class RemoteRuntimeServer
    : public registry::RegisteredType<RemoteRuntimeServer> {
public:
  // Initialization
  // This accepts a dictionary of key-value strings specific to subclass
  // implementations.
  virtual void
  init(const std::unordered_map<std::string, std::string> &configs) = 0;
  // Start the server.
  virtual void start() = 0;
  // Stop the server.
  virtual void stop() = 0;
  // Return the version of the server implementation.
  // It's defined by the Json payload version that it can handle.
  virtual std::pair<int, int> version() const = 0;
  // Handle incoming kernel execution requests.
  virtual void handleRequest(std::size_t reqId, ExecutionContext &io_context,
                             const std::string &backendSimName,
                             std::string_view ir, std::string_view kernelName,
                             void *kernelArgs, std::uint64_t argsSize,
                             std::size_t seed) = 0;
  // Handle incoming VQE requests
  virtual void handleVQERequest(std::size_t reqId,
                                cudaq::ExecutionContext &io_context,
                                const std::string &backendSimName,
                                std::string_view ir, cudaq::gradient *gradient,
                                cudaq::optimizer &optimizer, const int n_params,
                                std::string_view kernelName,
                                std::size_t seed) = 0;
  // Destructor
  virtual ~RemoteRuntimeServer() = default;
};

/// Base interface encapsulating a CUDA-Q runtime client, delegating
/// kernel execution to a remote server.
class RemoteRuntimeClient
    : public registry::RegisteredType<RemoteRuntimeClient> {
public:
  // Configure the client, e.g., address of the server.
  virtual void
  setConfig(const std::unordered_map<std::string, std::string> &configs) = 0;

  // Return the API version of the client implementation.
  // It defines the version number of the constructed payload and server
  // compatibility check/lookup.
  virtual int version() const = 0;

  // Reset the random seed sequence using for remote execution.
  // This is triggered by a random seed value being set in CUDA-Q runtime.
  virtual void resetRemoteRandomSeed(std::size_t seed) = 0;

  // Return the remote capabilities of the server.
  virtual RemoteCapabilities getRemoteCapabilities() const = 0;

  // Delegate/send kernel execution to a remote server.
  // Subclass will implement necessary transport-layer serialization and
  // communication protocols. The `ExecutionContext` will be updated in-place as
  // if this was a local execution.
  virtual bool
  sendRequest(mlir::MLIRContext &mlirContext, ExecutionContext &io_context,
              SerializedCodeExecutionContext *serializedCodeContext,
              cudaq::gradient *vqe_gradient, cudaq::optimizer *vqe_optimizer,
              const int vqe_n_params, const std::string &backendSimName,
              const std::string &kernelName, void (*kernelFunc)(void *),
              const void *kernelArgs, std::uint64_t argsSize,
              std::string *optionalErrorMsg = nullptr) = 0;
  // Destructor
  virtual ~RemoteRuntimeClient() = default;
};
} // namespace cudaq
