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
#include "cudaq.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"

using namespace mlir;

namespace cudaq {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class BaseRemoteSimulatorQPU : public cudaq::QPU {
protected:
  std::string m_simName;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  std::unique_ptr<MLIRContext> m_mlirContext;
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client;

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

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info(
        "BaseRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContextPtr =
        [&]() -> cudaq::ExecutionContext * {
      std::scoped_lock<std::mutex> lock(m_contextMutex);
      const auto iter = m_contexts.find(std::this_thread::get_id());
      if (iter == m_contexts.end())
        return nullptr;
      return iter->second;
    }();

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
    const bool requestOkay =
        m_client->sendRequest(*m_mlirContext, executionContext, m_simName, name,
                              kernelFunc, args, voidStarSize, &errorMsg);
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
} // namespace cudaq
