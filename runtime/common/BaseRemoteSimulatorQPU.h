/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ArgumentConversion.h"
#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/Resources.h"
#include "common/RuntimeMLIR.h"
#include "common/SerializedCodeExecutionContext.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

namespace cudaq {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class BaseRemoteSimulatorQPU : public QPU {
protected:
  std::string m_simName;
  std::unordered_map<std::thread::id, ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  std::unique_ptr<mlir::MLIRContext> m_mlirContext;
  std::unique_ptr<RemoteRuntimeClient> m_client;
  bool in_resource_estimation = false;
  static constexpr std::array<std::string_view, 1>
      DISALLOWED_EXECUTION_CONTEXT = {"tracer"};

  static constexpr bool isDisallowed(std::string_view context) {
    return std::any_of(DISALLOWED_EXECUTION_CONTEXT.begin(),
                       DISALLOWED_EXECUTION_CONTEXT.end(),
                       [context](std::string_view disallowed) {
                         return disallowed == context;
                       });
  }

  /// @brief Return a pointer to the execution context for this thread. It will
  /// return `nullptr` if it was not found in `m_contexts`.
  ExecutionContext *getExecutionContextForMyThread() {
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    const auto iter = m_contexts.find(std::this_thread::get_id());
    if (iter == m_contexts.end())
      return nullptr;
    return iter->second;
  }

public:
  BaseRemoteSimulatorQPU()
      : QPU(), m_client(registry::get<RemoteRuntimeClient>("rest")) {}

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
    auto parts = split(backend, ';');
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

  void enqueue(QuantumTask &task) override {
    CUDAQ_INFO("BaseRemoteSimulatorQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchVQE(const std::string &name, const void *kernelArgs,
                 gradient *gradient, const spin_op &H, optimizer &optimizer,
                 const int n_params, const std::size_t shots) override {
    ExecutionContext *executionContextPtr = getExecutionContextForMyThread();

    if (executionContextPtr && executionContextPtr->name == "tracer")
      return;

    auto ctx = std::make_unique<ExecutionContext>("observe", shots);
    ctx->kernelName = name;
    ctx->spin = spin_op::canonicalize(H);
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

  void launchKernel(const std::string &name,
                    const std::vector<void *> &rawArgs) override {
    [[maybe_unused]] auto dynamicResult = launchKernelImpl(
        name, nullptr, nullptr, 0, 0, &rawArgs, mlir::ModuleOp{});
  }

  KernelThunkResultType
  launchKernel(const std::string &name, KernelThunkType kernelFunc, void *args,
               std::uint64_t voidStarSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    // Remote simulation cannot deal with rawArgs. Drop them on the floor.
    return launchKernelImpl(name, kernelFunc, args, voidStarSize, resultOffset,
                            nullptr, mlir::ModuleOp{});
  }

  KernelThunkResultType launchModule(const std::string &name,
                                     mlir::ModuleOp module,
                                     const std::vector<void *> &rawArgs,
                                     mlir::Type resTy) override {
    if (resTy) {
      // Looks very much like launchKernel(string, vector<ptr>*).
      return launchKernelImpl(name, nullptr, rawArgs.back(), 0, 0, &rawArgs,
                              module);
    }
    // Looks very much like launchKernel(string, vector<ptr>*).
    return launchKernelImpl(name, nullptr, nullptr, 0, 0, &rawArgs, module);
  }

  [[nodiscard]] KernelThunkResultType launchKernelImpl(
      const std::string &name, KernelThunkType kernelFunc, void *args,
      std::uint64_t voidStarSize, std::uint64_t resultOffset,
      const std::vector<void *> *rawArgs, mlir::ModuleOp prefabMod) {
    CUDAQ_INFO("BaseRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
               "(simulator = {})",
               name, qpu_id, m_simName);

    if (in_resource_estimation)
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");

    ExecutionContext *executionContextPtr = getExecutionContextForMyThread();

    if (executionContextPtr && isDisallowed(executionContextPtr->name))
      throw std::runtime_error(
          executionContextPtr->name +
          " operation is not supported with cudaq target remote-mqpu!");

    // Run resource estimation locally
    if (executionContextPtr && executionContextPtr->name == "resource-count") {
      in_resource_estimation = true;
      getExecutionManager()->setExecutionContext(executionContextPtr);
      auto moduleOp = [&]() {
        if (prefabMod) {
          if (!rawArgs)
            throw std::runtime_error(
                "must provide launch arguments (got nullptr)");
          detail::mergeAllCallableClosures(prefabMod, name, *rawArgs);
          return m_client->lowerKernelInPlace(prefabMod, name, *rawArgs);
        }
        return m_client->lowerKernel(*m_mlirContext, name, args, voidStarSize,
                                     0, rawArgs);
      }();

      auto *jit = createQIRJITEngine(moduleOp, "qir-adaptive");

      auto funcPtr =
          jit->lookup(std::string(runtime::cudaqGenPrefixName) + name);
      if (!funcPtr)
        throw std::runtime_error(
            "cudaq::builder failed to get kernelReg function.");
      reinterpret_cast<void (*)()>(*funcPtr)();
      delete jit;
      getExecutionManager()->resetExecutionContext();
      in_resource_estimation = false;
      return {};
    }

    // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
    // was set before launching the kernel. Use a static variable per thread to
    // set up a single-shot execution context for this case.
    static thread_local ExecutionContext defaultContext("sample",
                                                        /*shots=*/1);
    // This is a kernel invocation outside the CUDA-Q APIs (sample/observe).
    const bool isDirectInvocation = !executionContextPtr;
    ExecutionContext &executionContext =
        executionContextPtr ? *executionContextPtr : defaultContext;

    // Populate the conditional feedback metadata if this is a direct
    // invocation (not otherwise populated by cudaq::sample)
    if (isDirectInvocation)
      executionContext.hasConditionalsOnMeasureResults =
          kernelHasConditionalFeedback(name);

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, executionContext, /*serializedCodeContext=*/nullptr,
        /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
        m_simName, name, make_degenerate_kernel_type(kernelFunc), args,
        voidStarSize, &errorMsg, rawArgs, prefabMod);
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

    // Assumes kernel has no dynamic results. (Static result handled above.)
    return {};
  }

  void launchSerializedCodeExecution(
      const std::string &name,
      SerializedCodeExecutionContext &serializeCodeExecutionObject) override {
    CUDAQ_INFO(
        "BaseRemoteSimulatorQPU: Launch remote code named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);

    ExecutionContext *executionContextPtr = getExecutionContextForMyThread();

    if (executionContextPtr && executionContextPtr->name == "tracer")
      return;

    // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
    // was set before launching the kernel. Use a static variable per thread to
    // set up a single-shot execution context for this case.
    static thread_local ExecutionContext defaultContext("sample",
                                                        /*shots=*/1);
    ExecutionContext &executionContext =
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

  void setExecutionContext(ExecutionContext *context) override {
    CUDAQ_INFO("BaseRemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts[std::this_thread::get_id()] = context;
  }

  void resetExecutionContext() override {
    CUDAQ_INFO("BaseRemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }

  void onRandomSeedSet(std::size_t seed) override {
    m_client->resetRemoteRandomSeed(seed);
  }
};

/// Implementation of base QPU subtype that submits simulation request to
/// NVCF.
class BaseNvcfSimulatorQPU : public BaseRemoteSimulatorQPU {
public:
  BaseNvcfSimulatorQPU() : BaseRemoteSimulatorQPU() {
    m_client = registry::get<RemoteRuntimeClient>("NVCF");
  }

  // Encapsulates Nvcf configurations that we need.
  // Empty strings mean no config available.
  struct NvcfConfig {
    std::string apiKey;
    std::string functionId;
    std::string versionId;
  };

  virtual void setTargetBackend(const std::string &backend) override {
    auto parts = split(backend, ';');
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
    if (fileExists(nvqcConfig)) {
      std::ifstream stream(nvqcConfig);
      std::string contents((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
      std::vector<std::string> lines;
      lines = split(contents, '\n');
      for (const std::string &l : lines) {
        std::vector<std::string> keyAndValue = split(l, ':');
        if (keyAndValue.size() != 2)
          throw std::runtime_error("Ill-formed configuration file (" +
                                   nvqcConfig +
                                   "). Key-value pairs must be in `<key> : "
                                   "<value>` format. (One per line)");
        trim(keyAndValue[0]);
        trim(keyAndValue[1]);
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
