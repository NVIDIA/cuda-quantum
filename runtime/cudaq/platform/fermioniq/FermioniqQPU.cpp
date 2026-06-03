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

cudaq::sample_result
cudaq::FermioniqQPU::launchKernel(const cudaq::sample_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] = compileKernelExecutions(policy, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

cudaq::async_sample_result
cudaq::FermioniqQPU::launchKernel(const cudaq::async_sample_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] =
      compileKernelExecutions(policy.inner, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.inner.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

cudaq::observe_result
cudaq::FermioniqQPU::launchKernel(const cudaq::observe_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] = compileKernelExecutions(policy, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  attachFermioniqObservable(codes[0], policy.spin);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

cudaq::async_observe_result
cudaq::FermioniqQPU::launchKernel(cudaq::async_observe_policy &policy,
                                  const AnyModule &module, KernelArgs args) {
  auto [kernelName, codes] =
      compileKernelExecutions(policy.inner, module, args);
  CUDAQ_INFO("FermioniqBaseQPU launching kernel ({}) with policy {}",
             kernelName, policy.inner.name);
  if (codes.size() != 1)
    throw std::runtime_error("Provider only allows 1 circuit at a time.");

  attachFermioniqObservable(codes[0], policy.inner.spin);
  return completeLaunchKernel(policy, kernelName, std::move(codes));
}

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::FermioniqQPU, fermioniq)
