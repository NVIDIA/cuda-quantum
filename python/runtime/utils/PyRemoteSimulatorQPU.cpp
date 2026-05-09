/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ArgumentWrapper.h"
#include "common/BaseRemoteSimulatorQPU.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include <mlir/IR/BuiltinOps.h>

using namespace mlir;

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU.
static void launchVqeImpl(cudaq::ExecutionContext *executionContextPtr,
                          std::unique_ptr<cudaq::RemoteRuntimeClient> &m_client,
                          const std::string &m_simName, const std::string &name,
                          const void *kernelArgs, cudaq::gradient *gradient,
                          const cudaq::spin_op &H, cudaq::optimizer &optimizer,
                          const int n_params, const std::size_t shots) {
  auto *wrapper = reinterpret_cast<const cudaq::ArgWrapper *>(kernelArgs);
  auto module = wrapper->mod;
  auto *mlirContext = module->getContext();

  if (executionContextPtr && executionContextPtr->name == "tracer")
    return;

  auto ctx = std::make_unique<cudaq::ExecutionContext>("observe", shots);
  ctx->kernelName = name;
  ctx->spin = cudaq::spin_op::canonicalize(H);
  if (shots > 0)
    ctx->shots = shots;

  std::string errorMsg;
  const bool requestOkay = m_client->sendRequest(
      *mlirContext, *executionContextPtr, gradient, &optimizer, n_params,
      m_simName, name, /*kernelFunc=*/nullptr, wrapper->rawArgs, /*argSize=*/0,
      &errorMsg);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch VQE. Error: " + errorMsg);
}

// This is a helper function to help reduce duplicated code across
// PyRemoteSimulatorQPU.
static void
launchKernelImpl(cudaq::ExecutionContext *executionContextPtr,
                 std::unique_ptr<cudaq::RemoteRuntimeClient> &remote_client,
                 const std::string &sim_name, const std::string &name,
                 void (*kernelFunc)(void *), cudaq::KernelArgs args) {
  const auto packed = args.getPacked();
  const auto rawArgs = args.getTypeErased();

  mlir::ModuleOp module;
  void *sendArgs = nullptr;
  std::uint64_t sendSize = 0;
  std::span<void *const> rawArgsForSend;

  if (kernelFunc) {
    if (!packed || packed->data.empty())
      throw std::runtime_error(
          "PyRemoteSimulatorQPU: launchKernel with a kernel thunk requires "
          "packed arguments.");
    auto *wrapper = reinterpret_cast<cudaq::ArgWrapper *>(packed->data.data());
    module = wrapper->mod;
    sendArgs = wrapper->rawArgs;
    sendSize = packed->data.size();
  } else {
    if (!rawArgs || rawArgs->empty())
      throw std::runtime_error(
          "Streamlined kernel launch: arguments cannot be empty. The first "
          "argument should be a pointer to the MLIR ModuleOp.");
    auto *moduleOpPtr = reinterpret_cast<mlir::ModuleOp *>((*rawArgs)[0]);
    module = *moduleOpPtr;
    // Remove the first argument (the MLIR ModuleOp) from the list of args.
    rawArgsForSend = rawArgs->subspan(1);
  }

  auto *mlirContext = module->getContext();

  // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
  // was set before launching the kernel. Use a static variable per thread to
  // set up a single-shot execution context for this case.
  static thread_local cudaq::ExecutionContext defaultContext("sample",
                                                             /*shots=*/1);
  cudaq::ExecutionContext &executionContext =
      executionContextPtr ? *executionContextPtr : defaultContext;
  std::string errorMsg;
  const bool requestOkay = remote_client->sendRequest(
      *mlirContext, executionContext,
      /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
      sim_name, name, kernelFunc, sendArgs, sendSize, &errorMsg,
      rawArgsForSend);
  if (!requestOkay)
    throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
}

template <typename Derived, typename Base>
class PyRemoteSimulatorCommonBase : public Base {
public:
  PyRemoteSimulatorCommonBase() : Base() {
    this->m_mlirContext = cudaq_internal::compiler::getOwningMLIRContext();
  }
  PyRemoteSimulatorCommonBase(PyRemoteSimulatorCommonBase &&) = delete;
  virtual ~PyRemoteSimulatorCommonBase() = default;

  bool isEmulated() override { return true; }

  void launchVQE(const std::string &name, const void *kernelArgs,
                 cudaq::gradient *gradient, const cudaq::spin_op &H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots) override {
    CUDAQ_INFO(
        "{}: Launch VQE kernel named '{}' remote QPU {} (simulator = {})",
        Derived::class_name, name, this->qpu_id, this->m_simName);
    ::launchVqeImpl(cudaq::getExecutionContext(), this->m_client,
                    this->m_simName, name, kernelArgs, gradient, H, optimizer,
                    n_params, shots);
  }

  cudaq::KernelThunkResultType launchKernel(const cudaq::SourceModule &src,
                                            cudaq::KernelArgs args) override {
    const auto &name = src.getName();
    auto rawFn = src.getFunctionPtr();
    cudaq::KernelThunkType kernelFunc = rawFn ? rawFn->getFn() : nullptr;
    if (kernelFunc) {
      CUDAQ_INFO("{}: Launch kernel named '{}' remote QPU {} (simulator = {})",
                 Derived::class_name, name, this->qpu_id, this->m_simName);
    } else {
      CUDAQ_INFO("{}: Streamline launch kernel named '{}' remote QPU {} "
                 "(simulator = {})",
                 Derived::class_name, name, this->qpu_id, this->m_simName);
    }
    ::launchKernelImpl(cudaq::getExecutionContext(), this->m_client,
                       this->m_simName, name,
                       make_degenerate_kernel_type(kernelFunc), args);
    return {};
  }
};

namespace {

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class PyRemoteSimulatorQPU
    : public PyRemoteSimulatorCommonBase<PyRemoteSimulatorQPU,
                                         cudaq::BaseRemoteSimulatorQPU> {
public:
  using Base = PyRemoteSimulatorCommonBase<PyRemoteSimulatorQPU,
                                           cudaq::BaseRemoteSimulatorQPU>;
  using Base::Base;
  virtual ~PyRemoteSimulatorQPU() = default;
  static constexpr const char class_name[] = "PyRemoteSimulatorQPU";
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::QPU, PyRemoteSimulatorQPU, RemoteSimulatorQPU)
