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

  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);
    launchImpl(kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
      return rawArgs.empty()
                 ? compiler.lowerQuakeCode(ctx, kernelName, args, {})
                 : compiler.lowerQuakeCode(ctx, kernelName, nullptr, rawArgs);
    });
    return {};
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    launchKernel(kernelName, nullptr, nullptr, 0, 0, rawArgs);
  }

  KernelThunkResultType launchModule(const std::string &kernelName,
                                     mlir::ModuleOp module,
                                     const std::vector<void *> &rawArgs,
                                     mlir::Type resTy) override {
    CUDAQ_INFO("FermioniqBaseQPU launching kernel via module ({})", kernelName);
    launchImpl(kernelName, [&](Compiler &compiler, ExecutionContext *ctx) {
      return compiler.lowerQuakeCode(ctx, kernelName, module, rawArgs);
    });
    return {};
  }

private:
  void
  launchImpl(const std::string &kernelName,
             std::function<std::vector<KernelExecution>(Compiler &,
                                                        ExecutionContext *)>
                 lower) {
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
    auto codes = lower(compiler, compileCtx);

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

    completeLaunchKernel(kernelName, std::move(codes));
  }
};
} // namespace cudaq
