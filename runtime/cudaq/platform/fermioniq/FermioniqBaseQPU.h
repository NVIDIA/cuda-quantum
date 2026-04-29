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
class FermioniqBaseQPU : public BaseRemoteRESTQPU {
public:
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

  KernelThunkResultType launchKernel(const SourceModule &src,
                                     KernelArgs args) override {
    const auto &kernelName = src.getName();
    CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);
    auto [module, context] = Compiler::loadQuakeCodeByName(kernelName);
    auto compiled =
        compileImpl(kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
          return compiler.runPassPipeline(ctx, kernelName, module, args,
                                          std::move(context));
        });
    launchImpl(compiled);
    return {};
  }

  KernelThunkResultType launchModule(const CompiledModule &compiled,
                                     KernelArgs args) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel via module ({})",
               compiled.getName());
    launchImpl(compiled);
    return {};
  }

  CompiledModule compileModule(const SourceModule &src, KernelArgs args,
                               bool isEntryPoint) override {
    const auto &kernelName = src.getName();
    auto mlirArt = src.getMlir();
    if (!mlirArt)
      throw std::runtime_error(
          "FermioniqBaseQPU::compileModule requires an MLIR artifact on the "
          "SourceModule for kernel '" +
          kernelName + "'.");
    auto module =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    CUDAQ_INFO("FermioniqBaseQPU compiling kernel via module ({})", kernelName);
    return compileImpl(
        kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
          return compiler.runPassPipeline(ctx, kernelName, module, args);
        });
  }

private:
  CompiledModule
  compileImpl(const std::string &kernelName,
              std::function<CompiledModule(Compiler &, ExecutionContext *)>
                  runPassPipeline) {
    auto *executionContext = getExecutionContext();
    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::contrib::draw().");

    // When the user issues an observe call, we don't want to use the default
    // CUDA-Q behaviour that splits up the circuit into several ansatz
    // sub circuit. Instead, we pass a "sample" context to the compiler to
    // prevent circuit splitting. This target handles observable evaluation
    // server-side.
    cudaq::ExecutionContext sampleContext("sample", 1);
    ExecutionContext *compileCtx = (executionContext->name == "observe")
                                       ? &sampleContext
                                       : executionContext;

    Compiler compiler(serverHelper.get(), backendConfig, targetConfig,
                      noiseModel, emulate);
    return runPassPipeline(compiler, compileCtx);
  }

  void launchImpl(const CompiledModule &compiled) {
    Compiler compiler(serverHelper.get(), backendConfig, targetConfig,
                      noiseModel, emulate);
    auto *executionContext = getExecutionContext();
    // TODO future iterations of this should support non-void return types.
    if (!executionContext)
      throw std::runtime_error(
          "Remote rest execution can only be performed via cudaq::sample(), "
          "cudaq::observe(), or cudaq::contrib::draw().");

    auto codes = compiler.emitKernelExecutions(compiled);

    if (codes.size() != 1)
      throw std::runtime_error("Provider only allows 1 circuit at a time.");

    if (executionContext->name == "observe") {
      auto spin = executionContext->spin.value();
      auto user_data = nlohmann::json::object();
      auto obs = nlohmann::json::array();
      for (const auto &term : spin) {
        auto terms = nlohmann::json::array();
        terms.push_back(term.get_term_id());
        auto coeff = term.evaluate_coefficient();
        auto coeff_str = cudaq_fmt::format("{}{}{}j", coeff.real(),
                                           coeff.imag() < 0.0 ? "-" : "+",
                                           std::fabs(coeff.imag()));
        terms.push_back(coeff_str);
        obs.push_back(terms);
      }
      user_data["observable"] = obs;
      codes[0].user_data = user_data;
    }

    completeLaunchKernel(compiled.getName(), std::move(codes));
  }
};
} // namespace cudaq
