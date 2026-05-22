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

  KernelThunkResultType unifiedLaunchModule(const AnyModule &module,
                                            KernelArgs args) override {
    if (std::holds_alternative<SourceModule>(module)) {
      const auto &src = std::get<SourceModule>(module);
      const auto &kernelName = src.getName();
      CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);
      auto [quakeModule, context] = Compiler::loadQuakeCodeByName(kernelName);
      auto compiled = compileImpl(
          kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
            return compiler.runPassPipeline(ctx, kernelName, quakeModule, args,
                                            std::move(context));
          });
      launchImpl(compiled);
    } else {
      const auto &compiled = std::get<CompiledModule>(module);
      CUDAQ_INFO("FermioniqBaseQPU launching kernel via module ({})",
                 compiled.getName());
      launchImpl(compiled);
    }
    return {};
  }

  CompiledModule compileModule(const SourceModule &src, KernelArgs args,
                               bool isEntryPoint) override {
    const auto &kernelName = src.getName();
    auto modulePtr = compileModulePreamble(src);
    CUDAQ_INFO("FermioniqBaseQPU compiling kernel via module ({})", kernelName);
    return compileImpl(
        kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
          return compiler.runPassPipeline(ctx, kernelName, modulePtr, args);
        });
  }

  sample_result launchKernel(sample_policy &policy, const AnyModule &module,
                             KernelArgs args) override;

  async_sample_result launchKernel(async_sample_policy &policy,
                                   const AnyModule &module,
                                   KernelArgs args) override;

  CompiledModule compileModule(sample_policy &policy, const SourceModule &src,
                               KernelArgs args, bool isEntryPoint) override {
    return compileModuleImpl(policy, src, args, isEntryPoint);
  }

private:
  CompiledModule
  compileImpl(const std::string &kernelName,
              std::function<CompiledModule(Compiler &, ExecutionContext *)>
                  runPassPipeline);

  template <typename Policy>
  CompiledModule compileImpl(Policy &policy, const SourceModule &src,
                             KernelArgs args, bool isEntryPoint) {
    const auto &kernelName = src.getName();
    auto modulePtr = compileModulePreamble(src);
    CUDAQ_INFO("FermioniqBaseQPU compiling kernel via module ({})", kernelName);
    return compileImpl(
        kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
          return compiler.runPassPipeline(policy, kernelName, modulePtr, args);
        });
  }
  void launchImpl(const CompiledModule &compiled);
};

} // namespace cudaq
