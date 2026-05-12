/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CompiledModule.h"
#include "common/DeviceCodeRegistry.h"
#include "common/ExecutionContext.h"
#include "common/RemoteKernelExecutor.h"
#include "common/Resources.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/algorithms/optimizer.h"
#include "cudaq/platform/platform_iface.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <fstream>

namespace cudaq {
class gradient;

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class BaseRemoteSimulatorQPU : public QPU {
protected:
  std::string m_simName;
  std::unique_ptr<mlir::MLIRContext> m_mlirContext;
  std::unique_ptr<RemoteRuntimeClient> m_client;
  bool in_resource_estimation = false;

public:
  BaseRemoteSimulatorQPU()
      : QPU(), m_client(registry::get<RemoteRuntimeClient>("rest")) {}

  BaseRemoteSimulatorQPU(BaseRemoteSimulatorQPU &&) = delete;
  virtual ~BaseRemoteSimulatorQPU() = default;

  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  // Get the capabilities from the client.
  virtual RemoteCapabilities getRemoteCapabilities() const override {
    return m_client->getRemoteCapabilities();
  }

  void
  configureExecutionContext(cudaq::ExecutionContext &context) const override {
    if (context.executionManager)
      context.executionManager->configureExecutionContext(context);
  }

  void
  finalizeExecutionContext(cudaq::ExecutionContext &context) const override {
    if (context.executionManager)
      context.executionManager->finalizeExecutionContext(context);
  }

  void beginExecution() override {
    auto executionContext = getExecutionContext();
    if (executionContext && executionContext->executionManager)
      executionContext->executionManager->beginExecution();
  }

  void endExecution() override {
    auto executionContext = getExecutionContext();
    if (executionContext && executionContext->executionManager)
      executionContext->executionManager->endExecution();
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
    ExecutionContext *executionContextPtr = getExecutionContext();

    if (executionContextPtr && executionContextPtr->name == "tracer")
      return;

    auto ctx = std::make_unique<ExecutionContext>("observe", shots);
    ctx->kernelName = name;
    ctx->spin = spin_op::canonicalize(H);
    if (shots > 0)
      ctx->shots = shots;

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, *executionContextPtr, gradient, &optimizer, n_params,
        m_simName, name, /*kernelFunc=*/nullptr, kernelArgs, /*argSize=*/0,
        &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch VQE. Error: " + errorMsg);
  }

  KernelThunkResultType unifiedLaunchModule(const AnyModule &module,
                                            KernelArgs args) override {
    if (std::holds_alternative<SourceModule>(module)) {
      const auto &src = std::get<SourceModule>(module);
      const auto &name = src.getName();
      auto rawFn = src.getFunctionPtr();
      KernelThunkType kernelFunc = rawFn ? rawFn->getFn() : nullptr;
      // Make sure at most one argument representation is present.
      KernelArgs forwarded;
      if (kernelFunc) {
        if (auto packed = args.getPacked())
          forwarded = KernelArgs{*packed};
      } else if (auto rawArgs = args.getTypeErased()) {
        forwarded = KernelArgs{*rawArgs};
      }
      auto compiled =
          compileKernelImpl(name, forwarded, mlir::ModuleOp{}, mlir::Type{});
      return launchKernelImpl(compiled, kernelFunc, forwarded);
    }

    const auto &compiled = std::get<CompiledModule>(module);
    auto rawArgs = args.getTypeErased();
    auto resultInfo = compiled.getResultInfo();
    void *resultBuf = nullptr;
    if (resultInfo.hasResult()) {
      assert(rawArgs && "no return buffer for kernel with result");
      resultBuf = rawArgs->back();
    }
    return launchKernelImpl(compiled, nullptr, args, resultBuf);
  }

  CompiledModule compileModule(const SourceModule &src, KernelArgs args,
                               bool isEntryPoint) override {
    const auto &kernelName = src.getName();
    auto mlirArt = src.getMlir();
    if (!mlirArt)
      throw std::runtime_error(
          "BaseRemoteSimulatorQPU::compileModule requires an MLIR artifact on "
          "the SourceModule for kernel '" +
          kernelName + "'.");
    auto module =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    CUDAQ_INFO("specializing remote simulator kernel via module ({})",
               kernelName);
    std::string fullName = cudaq::runtime::cudaqGenPrefixName + kernelName;
    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(fullName);
    auto resTy = cudaq::runtime::getReturnType(funcOp);
    return compileKernelImpl(kernelName, args, module, resTy);
  }

  [[nodiscard]] CompiledModule compileKernelImpl(const std::string &name,
                                                 KernelArgs args,
                                                 mlir::ModuleOp prefabMod,
                                                 mlir::Type resTy) {
    CUDAQ_INFO(
        "BaseRemoteSimulatorQPU: Compile kernel named '{}' remote QPU {} "
        "(simulator = {})",
        name, qpu_id, m_simName);

    if (in_resource_estimation)
      throw std::runtime_error(
          "Illegal use of resource counter simulator! (Did you attempt to run "
          "a kernel inside of a choice function?)");

    ExecutionContext *executionContextPtr = getExecutionContext();

    if (executionContextPtr && executionContextPtr->name == "tracer") {
      return cudaq_internal::compiler::CompiledModuleHelper::
          createCompiledModule(name, {}, {});
    }

    auto resultInfo =
        cudaq_internal::compiler::CompiledModuleHelper::createResultInfo(
            resTy, true, prefabMod);

    // Run resource estimation locally
    if (executionContextPtr && executionContextPtr->name == "resource-count") {
      in_resource_estimation = true;
      auto packed = args.getPacked();
      auto rawArgs = args.getTypeErased();
      auto moduleOp = [&]() {
        if (prefabMod) {
          if (!rawArgs)
            throw std::runtime_error(
                "must provide launch arguments (got nullptr)");
          cudaq_internal::compiler::mergeAllCallableClosures(prefabMod, name,
                                                             *rawArgs);
          return m_client->lowerKernelInPlace(prefabMod, name, *rawArgs);
        }
        return m_client->lowerKernel(
            *m_mlirContext, name, packed ? packed->data.data() : nullptr,
            packed ? packed->data.size() : 0, 0,
            rawArgs.value_or(std::span<void *const>{}));
      }();

      auto jit =
          cudaq_internal::compiler::createJITEngine(moduleOp, "qir-adaptive");
      auto artifacts =
          cudaq_internal::compiler::CompiledModuleHelper::createJitArtifacts(
              name, jit, {}, true);
      auto mlirArtifact =
          cudaq_internal::compiler::CompiledModuleHelper::createMlirArtifact(
              name, moduleOp);
      artifacts.push_back(mlirArtifact);
      return cudaq_internal::compiler::CompiledModuleHelper::
          createCompiledModule(name, resultInfo, std::move(artifacts));
    }

    auto mlirArtifact =
        cudaq_internal::compiler::CompiledModuleHelper::createMlirArtifact(
            name, prefabMod);

    return cudaq_internal::compiler::CompiledModuleHelper::createCompiledModule(
        name, resultInfo, {mlirArtifact});
  }

  [[nodiscard]] KernelThunkResultType
  launchKernelImpl(const CompiledModule &compiledModule,
                   KernelThunkType kernelFunc, KernelArgs args,
                   void *moduleResultBuf = nullptr) {
    auto name = compiledModule.getName();
    CUDAQ_INFO("BaseRemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
               "(simulator = {})",
               name, qpu_id, m_simName);

    auto packed = args.getPacked();
    auto rawArgs = args.getTypeErased();
    // Packed args: place result in packed buffer,
    // Type-erased args: place result in `moduleResultBuf`
    void *sendArgs = packed ? packed->data.data() : moduleResultBuf;
    std::uint64_t sendSize = packed ? packed->data.size() : 0;
    std::uint64_t resultOffset = packed ? packed->resultOffset : 0;

    ExecutionContext *executionContextPtr = getExecutionContext();

    if (in_resource_estimation) {
      auto jit = compiledModule.getJit();
      assert(jit.has_value());

      ExecutionContext ctx(executionContextPtr->name,
                           executionContextPtr->shots,
                           executionContextPtr->qpuId);
      ctx.kernelName = executionContextPtr->kernelName;
      ctx.executionManager = cudaq::getDefaultExecutionManager();
      cudaq::platform::with_execution_context(
          ctx, [jit, name]() { jit->getFn()(); });
      in_resource_estimation = false;
      return {};
    }

    auto mlir = compiledModule.getMlir();
    if (!mlir.has_value()) {
      assert(executionContextPtr && executionContextPtr->name == "tracer");
      return {};
    }

    // Default context for a 'fire-and-ignore' kernel launch; i.e., no context
    // was set before launching the kernel. Use a static variable per thread to
    // set up a single-shot execution context for this case.
    static thread_local ExecutionContext defaultContext("", /*shots=*/1);
    // This is a kernel invocation outside the CUDA-Q APIs (sample/observe).
    const bool isDirectInvocation = !executionContextPtr;
    ExecutionContext &executionContext =
        executionContextPtr ? *executionContextPtr : defaultContext;

    // Populate the conditional feedback metadata if this is a direct
    // invocation (not otherwise populated by cudaq::sample)
    if (isDirectInvocation)
      executionContext.hasConditionalsOnMeasureResults =
          kernelHasConditionalFeedback(name);

    auto moduleOp =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(*mlir);

    std::string errorMsg;
    const bool requestOkay = m_client->sendRequest(
        *m_mlirContext, executionContext,
        /*vqe_gradient=*/nullptr, /*vqe_optimizer=*/nullptr, /*vqe_n_params=*/0,
        m_simName, name, make_degenerate_kernel_type(kernelFunc), sendArgs,
        sendSize, &errorMsg, rawArgs.value_or(std::span<void *const>{}),
        moduleOp);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
    if (isDirectInvocation &&
        !executionContext.invocationResultBuffer.empty()) {
      if (executionContext.invocationResultBuffer.size() + resultOffset >
          sendSize)
        throw std::runtime_error(
            "Unexpected result: return type size of " +
            std::to_string(executionContext.invocationResultBuffer.size()) +
            " bytes overflows the argument buffer.");
      // Currently, we only support result buffer serialization on LittleEndian
      // CPUs (x86, ARM, PPC64LE).
      // If the client (e.g., compiled from source) is built for big-endian, we
      // will throw an error if result buffer data is returned.
      if (llvm::sys::IsBigEndianHost)
        throw std::runtime_error(
            "Serializing the result buffer from a remote kernel invocation is "
            "not supported for BigEndian CPU architectures.");

      char *resultBuf = reinterpret_cast<char *>(sendArgs) + resultOffset;
      // Copy the result data to the args buffer.
      std::memcpy(resultBuf, executionContext.invocationResultBuffer.data(),
                  executionContext.invocationResultBuffer.size());
      executionContext.invocationResultBuffer.clear();
    }

    // Assumes kernel has no dynamic results. (Static result handled above.)
    return {};
  }

  void onRandomSeedSet(std::size_t seed) override {
    m_client->resetRemoteRandomSeed(seed);
  }
};

} // namespace cudaq
