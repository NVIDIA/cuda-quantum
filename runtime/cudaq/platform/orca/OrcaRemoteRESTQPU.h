/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  /// @brief The number of shots
  std::optional<int> nShots;

  /// @brief the platform file path, CUDAQ_INSTALL/platforms
  std::filesystem::path platformPath;

  /// @brief The name of the QPU being targeted
  std::string qpuName;

  /// @brief Flag indicating whether we should emulate
  /// execution locally.
  bool emulate = false;

  /// @brief Pointer to the concrete Executor for this QPU
  std::unique_ptr<OrcaExecutor> executor;

  /// @brief Pointer to the concrete ServerHelper, provides
  /// specific JSON payloads and POST/GET URL paths.
  std::unique_ptr<ServerHelper> serverHelper;

  /// @brief Mapping of general key-values for backend
  /// configuration.
  std::map<std::string, std::string> backendConfig;

  /// @brief Mapping of thread and execution context
  std::unordered_map<std::size_t, cudaq::ExecutionContext *> contexts;

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

  /// @brief Get id of the thread this queue executes on.
  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  /// @brief Enqueue a quantum task on the asynchronous execution queue.
  void enqueue(cudaq::QuantumTask &task) override {
    CUDAQ_INFO("OrcaRemoteRESTQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  /// @brief Return true if the current backend is a simulator
  bool isSimulator() override { return emulate; }

  /// @brief Return true if the current backend supports conditional feedback
  bool supportsConditionalFeedback() override { return false; }

  /// @brief Return true if the current backend supports explicit measurements
  bool supportsExplicitMeasurements() override { return false; }

  /// @brief Provide the number of shots
  void setShots(int _nShots) override { nShots = _nShots; }

  /// @brief Clear the number of shots
  void clearShots() override { nShots = std::nullopt; }

  /// @brief Return true if the current backend is remote
  virtual bool isRemote() override { return !emulate; }

  /// @brief Store the execution context for launching kernel
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    CUDAQ_INFO("OrcaRemoteRESTQPU::setExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts.emplace(tid, context);
    cudaq::getExecutionManager()->setExecutionContext(contexts[tid]);
  }

  /// @brief Overrides resetExecutionContext
  void resetExecutionContext() override {
    CUDAQ_INFO("OrcaRemoteRESTQPU::resetExecutionContext QPU {}", qpu_id);
    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
    contexts[tid] = nullptr;
    contexts.erase(tid);
  }

  /// @brief This setTargetBackend override is in charge of reading the
  /// specific target backend configuration file.
  void setTargetBackend(const std::string &backend) override;

  [[nodiscard]] KernelThunkResultType
  launchKernelCommon(const std::string &kernelName, KernelThunkType kernelFunc,
                     void *args);

  /// @brief Launch the kernel. Handle all pertinent modifications for the
  /// execution context.
  [[nodiscard]] KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    return launchKernelCommon(kernelName, kernelFunc, args);
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override;
};
} // namespace cudaq
