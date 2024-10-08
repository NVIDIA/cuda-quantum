/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "OrcaExecutor.h"
#include "common/ExecutionContext.h"
#include "common/Future.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/platform/qpu.h"
#include "orca_qpu.h"

namespace cudaq {

/// @brief The OrcaRemoteRESTQPU is a subtype of QPU that enables the
/// execution of CUDA-Q kernels on remotely hosted quantum computing
/// services via a REST Client / Server interaction. This type is meant
/// to be general enough to support any remotely hosted service.
/// Moreover, this QPU handles launching kernels under the Execution Context
/// that includes sampling via synchronous client invocations.
class OrcaRemoteRESTQPU : public cudaq::QPU {
protected:
  /// The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Flag indicating whether we should emulate
  /// execution locally.
  bool emulate = false;

  // Pointer to the concrete Executor for this QPU
  std::unique_ptr<OrcaExecutor> executor;

  /// @brief Pointer to the concrete ServerHelper, provides
  /// specific JSON payloads and POST/GET URL paths.
  std::unique_ptr<ServerHelper> serverHelper;

  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

private:
  /// @brief RestClient used for HTTP requests.
  RestClient client;

public:
  /// @brief The constructor
  OrcaRemoteRESTQPU() : QPU() {
    std::filesystem::path cudaqLibPath{getCUDAQLibraryPath()};
    platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    // Default is to run sampling via the remote rest call
    executor = std::make_unique<OrcaExecutor>();
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

  /// @brief Launch the kernel. Handle all pertinent modifications for the
  /// execution context.
  void launchKernel(const std::string &kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset,
                    const std::vector<void *> &rawArgs) override;
  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    throw std::runtime_error("launch kernel on raw args not implemented");
  }
};
} // namespace cudaq
