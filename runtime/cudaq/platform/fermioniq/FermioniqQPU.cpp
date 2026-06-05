/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqQPU.h"
#include "nlohmann/json.hpp"
#include "cudaq/runtime/logger/cudaq_fmt.h"
#include <memory>
#include <optional>

cudaq::FermioniqQPU::~FermioniqQPU() = default;

cudaq::KernelThunkResultType
cudaq::FermioniqQPU::unifiedLaunchModule(const AnyModule &module,
                                         KernelArgs args) {
  auto *executionContext = getExecutionContext();
  Compiler compiler(getCompileTarget(executionContext));
  std::optional<CompiledModule> compiled;

  if (std::holds_alternative<SourceModule>(module)) {
    const auto &src = std::get<SourceModule>(module);
    const auto &kernelName = src.getName();
    CUDAQ_INFO("FermioniqBaseQPU launching kernel ({})", kernelName);
    auto [quakeModule, context] = Compiler::loadQuakeCodeByName(kernelName);
    compiled = compiler.runPassPipeline(kernelName, quakeModule, args, true,
                                        std::move(context));
  } else {
    compiled = std::get<CompiledModule>(module);
    CUDAQ_INFO("FermioniqBaseQPU launching kernel via module ({})",
               compiled->getName());
  }

  auto codes = compiler.emitKernelExecutions(*compiled);

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

  // Propagate metadata from the compiled artifact to the execution context.
  if (auto ctx = getExecutionContext()) {
    ctx->hasConditionalsOnMeasureResults =
        compiled->getMetadata().hasConditionalsOnMeasureResults;
  }

  completeLaunchKernel(compiled->getName(), std::move(codes));
  return {};
}

cudaq::sample_result
cudaq::FermioniqQPU::launchKernel(cudaq::sample_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] = compileKernelExecutions(policy, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

cudaq::async_sample_result
cudaq::FermioniqQPU::launchKernel(cudaq::async_sample_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] =
      compileKernelExecutions(policy.inner, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.inner.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::FermioniqQPU, fermioniq)
