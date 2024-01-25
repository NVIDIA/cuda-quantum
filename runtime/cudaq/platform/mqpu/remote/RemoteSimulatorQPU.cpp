/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"

#include <fstream>

namespace {
using namespace mlir;

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::QPU {
private:
  std::string m_simName;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  std::unique_ptr<MLIRContext> m_mlirContext;
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client;

public:
  RemoteSimulatorQPU()
      : QPU(), m_mlirContext(cudaq::initializeMLIR()),
        m_client(cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest")) {}

  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  virtual void setTargetBackend(const std::string &backend) override {
    auto parts = cudaq::split(backend, ';');
    if (parts.size() % 2 != 0)
      throw std::invalid_argument("Unexpected backend configuration string. "
                                  "Expecting a ';'-separated key-value pairs.");
    for (std::size_t i = 0; i < parts.size(); i += 2) {
      if (parts[i] == "target") {
        if (parts[i + 1] == "nvcf" || parts[i + 1] == "NVCF")
          m_client = cudaq::registry::get<cudaq::RemoteRuntimeClient>("NVCF");
        const std::string nvcfApiKey = searchAPIKey();
        m_client->setConfig({{"api-key", nvcfApiKey}});
      }
      if (parts[i] == "url")
        m_client->setConfig({{"url", parts[i + 1]}});
      if (parts[i] == "simulator")
        m_simName = parts[i + 1];
    }
  }

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("RemoteSimulatorQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("RemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
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
    cudaq::info("RemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts[std::this_thread::get_id()] = context;
  }

  void resetExecutionContext() override {
    cudaq::info("RemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }
  private:
  std::string searchAPIKey(const std::string &userSpecifiedConfigFile = "") {
    std::string hwConfig;
    // Allow someone to tweak this with an environment variable
    if (auto creds = std::getenv("CUDAQ_NVCF_CREDENTIALS"))
      hwConfig = std::string(creds);
    else if (!userSpecifiedConfigFile.empty())
      hwConfig = userSpecifiedConfigFile;
    else
      hwConfig = std::string(getenv("HOME")) + std::string("/.nvcf_config");
    if (cudaq::fileExists(hwConfig)) {
      std::ifstream stream(hwConfig);
      std::string contents((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
      std::vector<std::string> lines;
      lines = cudaq::split(contents, '\n');
      for (const std::string &l : lines) {
        std::vector<std::string> keyAndValue = cudaq::split(l, ':');
        if (keyAndValue.size() != 2)
          throw std::runtime_error("Ill-formed configuration file (" +
                                   hwConfig +
                                   "). Key-value pairs must be in `<key> : "
                                   "<value>` format. (One per line)");
        cudaq::trim(keyAndValue[0]);
        cudaq::trim(keyAndValue[1]);
        if (keyAndValue[0] == "key" || keyAndValue[0] == "apikey")
          return keyAndValue[1];
      }
    }

    throw std::runtime_error("Cannot find NVCF Config file with credentials "
                             "(~/.nvcf_config).");
    return "";
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, RemoteSimulatorQPU, RemoteSimulatorQPU)
