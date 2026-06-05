/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "FermioniqQPU.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "nlohmann/json.hpp"
#include "cudaq/runtime/logger/cudaq_fmt.h"

namespace {
void attachFermioniqObservable(cudaq::KernelExecution &code,
                               const cudaq::spin_op &spin) {
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
  code.user_data = user_data;
}
} // namespace

cudaq::FermioniqQPU::~FermioniqQPU() = default;

cudaq::observe_result
cudaq::FermioniqQPU::launchKernel(const cudaq::observe_policy &policy,
                                  const CompiledModule &module,
                                  KernelArgs args) {
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             module.getName(), policy.name);

  // TODO: This should be moved into compiler::compileModule, but this would add
  // a dependency on the compiler in the C++ launch path.
  auto compiled = module;
  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy));
  cudaq_internal::compiler::CompiledModuleHelper::ensureMlirArtifactsExist(
      compiled, compiler, args);

  auto codes = compiler.emitKernelExecutions(compiled);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  attachFermioniqObservable(codes[0], policy.spin);
  auto result =
      completeLaunchKernel(policy, compiled.getName(), std::move(codes));
  auto expectation = result.raw_data().expectation(GlobalRegisterName);
  return cudaq::observe_result(expectation, result.get_spin(),
                               result.raw_data());
}

cudaq::async_observe_result
cudaq::FermioniqQPU::launchKernel(cudaq::async_observe_policy &policy,
                                  const CompiledModule &module,
                                  KernelArgs args) {
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             module.getName(), policy.inner.name);

  // TODO: This should be moved into compiler::compileModule, but this would add
  // a dependency on the compiler in the C++ launch path.
  auto compiled = module;
  cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy.inner));
  cudaq_internal::compiler::CompiledModuleHelper::ensureMlirArtifactsExist(
      compiled, compiler, args);

  auto codes = compiler.emitKernelExecutions(compiled);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  attachFermioniqObservable(codes[0], policy.inner.spin);
  return completeLaunchKernel(policy, compiled.getName(), std::move(codes));
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::FermioniqQPU, fermioniq)
