/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteSimulatorQPU.h"
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

namespace {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class PyRemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
public:
  PyRemoteSimulatorQPU() : BaseRemoteSimulatorQPU() {}

  virtual bool isEmulated() override { return true; }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("PyRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);
    auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(args);
    auto m_module = wrapper->mod;
    auto callableNames = wrapper->callableNames;

    auto *mlirContext = m_module->getContext();

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
    const bool requestOkay = m_client->sendRequest(
        *mlirContext, executionContext, m_simName, name, kernelFunc,
        wrapper->rawArgs, voidStarSize, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
  }

  PyRemoteSimulatorQPU(PyRemoteSimulatorQPU &&) = delete;
  virtual ~PyRemoteSimulatorQPU() = default;
};

/// Implementation of QPU subtype that submits simulation request to NVCF.
/// NOTE: This class duplicates the `isEmulated` and `launchKernel` methods from
/// `PyRemoteSimulatorQPU` class above; tried using multiple inheritance, but,
/// got errors from the functionality to register type.
class PyNvcfSimulatorQPU : public cudaq::BaseNvcfSimulatorQPU {
public:
  PyNvcfSimulatorQPU() : BaseNvcfSimulatorQPU() {}

  virtual bool isEmulated() override { return true; }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("PyNvcfSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);
    auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(args);
    auto m_module = wrapper->mod;
    auto callableNames = wrapper->callableNames;

    auto *mlirContext = m_module->getContext();

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
    const bool requestOkay = m_client->sendRequest(
        *mlirContext, executionContext, m_simName, name, kernelFunc,
        wrapper->rawArgs, voidStarSize, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
  }

  PyNvcfSimulatorQPU(PyNvcfSimulatorQPU &&) = delete;
  virtual ~PyNvcfSimulatorQPU() = default;
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, PyRemoteSimulatorQPU, RemoteSimulatorQPU)
CUDAQ_REGISTER_TYPE(cudaq::QPU, PyNvcfSimulatorQPU, NvcfSimulatorQPU)
