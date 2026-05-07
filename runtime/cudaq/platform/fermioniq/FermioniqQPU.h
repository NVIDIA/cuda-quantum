/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The `FermioniqBaseQPU` is a QPU that allows users to
// submit kernels to the Fermioniq simulator.
class FermioniqQPU : public BaseRemoteRESTQPU {
public:
  ~FermioniqQPU() override;

  virtual bool isRemote() override { return true; }

  /// @brief Return true if locally emulating a remote QPU
  virtual bool isEmulated() override { return false; }

  /// @brief Set the noise model, only allow this for
  /// emulation.
  virtual void setNoiseModel(const cudaq::noise_model *model) override {
    if (model) {
      throw std::runtime_error("Noise modeling is not allowed on this backend");
    }
  }

  /// Reset the execution context
  virtual void
  finalizeExecutionContext(ExecutionContext &context) const override {
    // set the pre-computed expectation value.
    if (context.name == "observe") {
      auto expectation = context.result.expectation(GlobalRegisterName);
      context.expectationValue = expectation;
    }
  }

  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);
    auto [module, context] = Compiler::loadQuakeCodeByName(kernelName);
    auto compiled = compileImpl(kernelName, [&](Compiler &compiler,
                                                ExecutionContext *ctx) {
      return rawArgs.empty()
                 ? compiler.runPassPipeline(ctx, kernelName, module, {}, args,
                                            std::move(context))
                 : compiler.runPassPipeline(ctx, kernelName, module, rawArgs,
                                            nullptr, std::move(context));
    });
    launchImpl(compiled);
    return {};
  }

  KernelThunkResultType
  launchModule(const CompiledModule &compiled,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel via module ({})",
               compiled.getName());
    launchImpl(compiled);
    return {};
  }

  CompiledModule compileModule(const std::string &kernelName,
                               const void *modulePtr,
                               const std::vector<void *> &rawArgs,
                               bool isEntryPoint) override {
    CUDAQ_INFO("FermioniqBaseQPU compiling kernel via module ({})", kernelName);
    return compileImpl(kernelName,
                       [&](Compiler &compiler, ExecutionContext *ctx) {
                         return compiler.runPassPipeline(
                             ctx, kernelName, modulePtr, rawArgs, nullptr);
                       });
  }

private:
  CompiledModule
  compileImpl(const std::string &kernelName,
              std::function<CompiledModule(Compiler &, ExecutionContext *)>
                  runPassPipeline);

  void launchImpl(const CompiledModule &compiled);
};

} // namespace cudaq
