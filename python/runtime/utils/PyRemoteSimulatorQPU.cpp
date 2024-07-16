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

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU and PyNvcfSimulatorQPU.
static void _launchVQE(cudaq::ExecutionContext *executionContextPtr,
                       std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
                       const std::string &m_simName, const std::string &name,
                       const void *kernelArgs, cudaq::gradient *gradient,
                       cudaq::spin_op &H, cudaq::optimizer &optimizer,
                       const int n_params, const std::size_t shots) {
  auto *wrapper = reinterpret_cast<const cudaq::ArgWrapper *>(kernelArgs);
  auto m_module = wrapper->mod;
  auto *mlirContext = m_module->getContext();

  if (executionContextPtr && executionContextPtr->name == "tracer")
    return;

  auto ctx = std::make_unique<cudaq::ExecutionContext>("observe", shots);
  ctx->kernelName = name;
  ctx->spin = &H;
  if (shots > 0)
    ctx->shots = shots;

  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(
      *mlirContext, *executionContextPtr, /*serializedCodeContext=*/nullptr,
      gradient, &optimizer, n_params, m_simName, name, /*kernelFunc=*/nullptr,
      wrapper->rawArgs, /*argSize=*/0, &errorMsg);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch VQE. Error: " + errorMsg);
}

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU and PyNvcfSimulatorQPU.
static void _launchKernel(cudaq::ExecutionContext *executionContextPtr,
                          std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
                          const std::string &m_simName, const std::string &name,
                          void (*kernelFunc)(void *), void *args,
                          std::uint64_t voidStarSize,
                          std::uint64_t resultOffset) {
  auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(args);
  auto m_module = wrapper->mod;
  auto callableNames = wrapper->callableNames;

  auto *mlirContext = m_module->getContext();

  // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
  // was set before launching the kernel. Use a static variable per thread to
  // set up a single-shot execution context for this case.
  static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                             /*shots=*/1);
  cudaq::ExecutionContext &executionContext =
      executionContextPtr ? *executionContextPtr : defaultContext;
  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(
      *mlirContext, executionContext, /*serializedCodeContext=*/nullptr,
      /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
      m_simName, name, kernelFunc, wrapper->rawArgs, voidStarSize, &errorMsg);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
}

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class PyRemoteSimulatorQPU : public cudaq::BaseRemoteSimulatorQPU {
public:
  PyRemoteSimulatorQPU() : BaseRemoteSimulatorQPU() {}

  virtual bool isEmulated() override { return true; }

  void launchVQE(const std::string &name, const void *kernelArgs,
                 cudaq::gradient *gradient, cudaq::spin_op H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots) override {
    cudaq::info(
        "PyRemoteSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);
    ::_launchVQE(getExecutionContextForMyThread(), m_client, m_simName, name,
                 kernelArgs, gradient, H, optimizer, n_params, shots);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("PyRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);
    ::_launchKernel(getExecutionContextForMyThread(), m_client, m_simName, name,
                    kernelFunc, args, voidStarSize, resultOffset);
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

  void launchVQE(const std::string &name, const void *kernelArgs,
                 cudaq::gradient *gradient, cudaq::spin_op H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots) override {
    cudaq::info(
        "PyNvcfSimulatorQPU: Launch VQE kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);
    ::_launchVQE(getExecutionContextForMyThread(), m_client, m_simName, name,
                 kernelArgs, gradient, H, optimizer, n_params, shots);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("PyNvcfSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);
    ::_launchKernel(getExecutionContextForMyThread(), m_client, m_simName, name,
                    kernelFunc, args, voidStarSize, resultOffset);
  }

  PyNvcfSimulatorQPU(PyNvcfSimulatorQPU &&) = delete;
  virtual ~PyNvcfSimulatorQPU() = default;
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, PyRemoteSimulatorQPU, RemoteSimulatorQPU)
CUDAQ_REGISTER_TYPE(cudaq::QPU, PyNvcfSimulatorQPU, NvcfSimulatorQPU)
