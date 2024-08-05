/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RuntimeMLIR.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq.h"
#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include <fstream>

namespace cudaq {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class BaseRemoteSimulatorQPU : public cudaq::QPU {
protected:
  std::string m_simName;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  std::unique_ptr<mlir::MLIRContext> m_mlirContext;
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client;

  /// @brief Return a pointer to the execution context for this thread. It will
  /// return `nullptr` if it was not found in `m_contexts`.
  cudaq::ExecutionContext *getExecutionContextForMyThread() {
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    const auto iter = m_contexts.find(std::this_thread::get_id());
    if (iter == m_contexts.end())
      return nullptr;
    return iter->second;
  }

public:
  BaseRemoteSimulatorQPU()
      : QPU(),
        m_client(cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest")) {}

  BaseRemoteSimulatorQPU(BaseRemoteSimulatorQPU &&) = delete;
  virtual ~BaseRemoteSimulatorQPU() = default;

  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  // Conditional feedback is handled by the server side.
  virtual bool supportsConditionalFeedback() override { return true; }

  // Get the capabilities from the client.
  virtual RemoteCapabilities getRemoteCapabilities() const override {
    return m_client->getRemoteCapabilities();
  }

  virtual void setTargetBackend(const std::string &backend) override {
    auto parts = cudaq::split(backend, ';');
    if (parts.size() % 2 != 0)
      throw std::invalid_argument("Unexpected backend configuration string. "
                                  "Expecting a ';'-separated key-value pairs.");
    for (std::size_t i = 0; i < parts.size(); i += 2) {
      if (parts[i] == "url")
        m_client->setConfig({{"url", parts[i + 1]}});
      if (parts[i] == "simulator")
        m_simName = parts[i + 1];
    }
  }

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("BaseRemoteSimulatorQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchVQE(const std::string &name, const void *kernelArgs,
                 cudaq::gradient *gradient, cudaq::spin_op H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots) override {
    cudaq::ExecutionContext *executionContextPtr =
        getExecutionContextForMyThread();

    if (executionContextPtr && executionContextPtr->name == "tracer")
      return;

    auto ctx = std::make_unique<ExecutionContext>("observe", shots);
    ctx->kernelName = name;
    ctx->spin = &H;
    if (shots > 0)
      ctx->shots = shots;

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, *executionContextPtr, /*serializedCodeContext=*/nullptr,
        gradient, &optimizer, n_params, m_simName, name, /*kernelFunc=*/nullptr,
        kernelArgs, /*argSize=*/0, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch VQE. Error: " + errorMsg);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info(
        "BaseRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContextPtr =
        getExecutionContextForMyThread();

    if (executionContextPtr && executionContextPtr->name == "tracer") {
      return;
    }

    // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
    // was set before launching the kernel. Use a static variable per thread to
    // set up a single-shot execution context for this case.
    static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                               /*shots=*/1);
    // This is a kernel invocation outside the CUDA-Q APIs (sample/observe).
    const bool isDirectInvocation = !executionContextPtr;
    cudaq::ExecutionContext &executionContext =
        executionContextPtr ? *executionContextPtr : defaultContext;

    // Populate the conditional feedback metadata if this is a direct
    // invocation (not otherwise populated by cudaq::sample)
    if (isDirectInvocation)
      executionContext.hasConditionalsOnMeasureResults =
          cudaq::kernelHasConditionalFeedback(name);

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, executionContext, /*serializedCodeContext=*/nullptr,
        /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
        m_simName, name, kernelFunc, args, voidStarSize, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
    if (isDirectInvocation &&
        !executionContext.invocationResultBuffer.empty()) {
      if (executionContext.invocationResultBuffer.size() + resultOffset >
          voidStarSize)
        throw std::runtime_error(
            "Unexpected result: return type size of " +
            std::to_string(executionContext.invocationResultBuffer.size()) +
            " bytes overflows the argument buffer.");
      // Currently, we only support result buffer serialization on LittleEndian
      // CPUs (x86, ARM, PPC64LE).
      // Note: NVQC service will always be using LE. If
      // the client (e.g., compiled from source) is built for big-endian, we
      // will throw an error if result buffer data is returned.
      if (llvm::sys::IsBigEndianHost)
        throw std::runtime_error(
            "Serializing the result buffer from a remote kernel invocation is "
            "not supported for BigEndian CPU architectures.");

      char *resultBuf = reinterpret_cast<char *>(args) + resultOffset;
      // Copy the result data to the args buffer.
      std::memcpy(resultBuf, executionContext.invocationResultBuffer.data(),
                  executionContext.invocationResultBuffer.size());
      executionContext.invocationResultBuffer.clear();
    }
  }

  void
  launchSerializedCodeExecution(const std::string &name,
                                cudaq::SerializedCodeExecutionContext
                                    &serializeCodeExecutionObject) override {
    cudaq::info(
        "BaseRemoteSimulatorQPU: Launch remote code named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContextPtr =
        getExecutionContextForMyThread();

    if (executionContextPtr && executionContextPtr->name == "tracer") {
      return;
    }

    // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
    // was set before launching the kernel. Use a static variable per thread to
    // set up a single-shot execution context for this case.
    static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                               /*shots=*/1);
    cudaq::ExecutionContext &executionContext =
        executionContextPtr ? *executionContextPtr : defaultContext;

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, executionContext, &serializeCodeExecutionObject,
        /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
        m_simName, name, /*kernelFunc=*/nullptr, /*args=*/nullptr,
        /*voidStarSize=*/0, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::info("BaseRemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts[std::this_thread::get_id()] = context;
  }

  void resetExecutionContext() override {
    cudaq::info("BaseRemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }

  void onRandomSeedSet(std::size_t seed) override {
    m_client->resetRemoteRandomSeed(seed);
  }
};

/// Implementation of base QPU subtype that submits simulation request to
/// NVCF.
class BaseNvcfSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
public:
  BaseNvcfSimulatorQPU() : BaseRemoteSimulatorQPU() {
    m_client = cudaq::registry::get<cudaq::RemoteRuntimeClient>("NVCF");
  }

  // Encapsulates Nvcf configurations that we need.
  // Empty strings mean no config available.
  struct NvcfConfig {
    std::string apiKey;
    std::string functionId;
    std::string versionId;
  };

  virtual void setTargetBackend(const std::string &backend) override {
    auto parts = cudaq::split(backend, ';');
    if (parts.size() % 2 != 0)
      throw std::invalid_argument("Unexpected backend configuration string. "
                                  "Expecting a ';'-separated key-value pairs.");
    std::string apiKey, functionId, versionId, ngpus;

    for (std::size_t i = 0; i < parts.size(); i += 2) {
      if (parts[i] == "simulator")
        m_simName = parts[i + 1];
      // First, check if api key or function Id is provided as target options.
      if (parts[i] == "function_id")
        functionId = parts[i + 1];
      if (parts[i] == "api_key")
        apiKey = parts[i + 1];
      if (parts[i] == "version_id")
        versionId = parts[i + 1];
      if (parts[i] == "ngpus")
        ngpus = parts[i + 1];
    }
    // If none provided, look for them in environment variables or the config
    // file.
    const auto config = searchNvcfConfig();
    if (apiKey.empty())
      apiKey = config.apiKey;
    if (functionId.empty())
      functionId = config.functionId;
    if (versionId.empty())
      versionId = config.versionId;

    // API key and function Id are required.
    if (apiKey.empty())
      throw std::runtime_error(
          "Cannot find NVQC API key. Please refer to the documentation for "
          "information about obtaining and using your NVQC API key.");

    if (!apiKey.starts_with("nvapi-"))
      std::runtime_error(
          "An invalid NVQC API key is provided. Please check your settings.");
    std::unordered_map<std::string, std::string> clientConfigs{
        {"api-key", apiKey}};
    if (!functionId.empty())
      clientConfigs.emplace("function-id", functionId);
    if (!versionId.empty())
      clientConfigs.emplace("version-id", versionId);
    if (!ngpus.empty())
      clientConfigs.emplace("ngpus", ngpus);

    m_client->setConfig(clientConfigs);
  }

  // The NVCF version of this function needs to dynamically fetch the remote
  // capabilities from the currently deployed servers.
  virtual RemoteCapabilities getRemoteCapabilities() const override {
    return m_client->getRemoteCapabilities();
  }

protected:
  // Helper to search NVQC config from environment variable or config file.
  NvcfConfig searchNvcfConfig() {
    NvcfConfig config;
    // Search from environment variable
    if (auto apiKey = std::getenv("NVQC_API_KEY"))
      config.apiKey = std::string(apiKey);

    if (auto funcIdEnv = std::getenv("NVQC_FUNCTION_ID"))
      config.functionId = std::string(funcIdEnv);

    if (auto versionIdEnv = std::getenv("NVQC_FUNCTION_VERSION_ID"))
      config.versionId = std::string(versionIdEnv);

    std::string nvqcConfig;
    // Allow someone to tweak this with an environment variable
    if (auto creds = std::getenv("CUDAQ_NVQC_CREDENTIALS"))
      nvqcConfig = std::string(creds);
    else
      nvqcConfig = std::string(getenv("HOME")) + std::string("/.nvqc_config");
    if (cudaq::fileExists(nvqcConfig)) {
      std::ifstream stream(nvqcConfig);
      std::string contents((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
      std::vector<std::string> lines;
      lines = cudaq::split(contents, '\n');
      for (const std::string &l : lines) {
        std::vector<std::string> keyAndValue = cudaq::split(l, ':');
        if (keyAndValue.size() != 2)
          throw std::runtime_error("Ill-formed configuration file (" +
                                   nvqcConfig +
                                   "). Key-value pairs must be in `<key> : "
                                   "<value>` format. (One per line)");
        cudaq::trim(keyAndValue[0]);
        cudaq::trim(keyAndValue[1]);
        if (config.apiKey.empty() &&
            (keyAndValue[0] == "key" || keyAndValue[0] == "apikey"))
          config.apiKey = keyAndValue[1];
        if (config.functionId.empty() && (keyAndValue[0] == "function-id" ||
                                          keyAndValue[0] == "Function ID"))
          config.functionId = keyAndValue[1];
        if (config.versionId.empty() &&
            (keyAndValue[0] == "version-id" || keyAndValue[0] == "Version ID"))
          config.versionId = keyAndValue[1];
      }
    }
    return config;
  }
};

} // namespace cudaq
