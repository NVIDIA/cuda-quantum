/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqQPU.h"
#include "nlohmann/json.hpp"
#include <memory>

cudaq::FermioniqQPU::~FermioniqQPU() = default;

cudaq::CompiledModule cudaq::FermioniqQPU::compileImpl(
    const std::string &kernelName,
    std::function<cudaq::CompiledModule(cudaq_internal::compiler::Compiler &,
                                        cudaq::ExecutionContext *)>
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
  ExecutionContext *compileCtx =
      (executionContext->name == "observe") ? &sampleContext : executionContext;

  Compiler compiler(serverHelper.get(), backendConfig, targetConfig, noiseModel,
                    emulate);
  return runPassPipeline(compiler, compileCtx);
}

void cudaq::FermioniqQPU::launchImpl(const cudaq::CompiledModule &compiled) {
  Compiler compiler(serverHelper.get(), backendConfig, targetConfig, noiseModel,
                    emulate);
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

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::FermioniqQPU, fermioniq)
