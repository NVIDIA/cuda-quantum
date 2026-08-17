/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"
#include "cudaq/platform/qpu_utils.h"
#include <optional>

namespace cudaq {

/// @brief Base QPU class for analog platforms like `quera` and `pasqal`.
/// Provides common functionality and implementation.
class AnalogRemoteRESTQPU : public BaseRemoteRESTQPU {
public:
  /// @brief Check if this is a remote target
  virtual bool isRemote() override { return true; }

  /// @brief Check if this is an emulated target
  virtual bool isEmulated() override { return false; }

  using BaseRemoteRESTQPU::getCompileTarget;
  using BaseRemoteRESTQPU::launchKernel;

  CompileTarget getCompileTarget(const sample_policy &) override {
    CompileTarget target;
    target.overrideAOTCompilation = false;
    return target;
  }

  /// @brief Launch a kernel with the given arguments
  /// Only analog Hamiltonian kernels are supported
  detail::future launchKernelCommon(const sample_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) {
    const auto &kernelName = module.getName();
    if (!cudaq::detail::isAnalogHamiltonianKernel(kernelName))
      throw std::runtime_error(
          "Arbitrary kernel execution is not supported on this target.");

    if (emulate)
      throw std::runtime_error(
          "Local emulation is not yet supported on this target.");

    CUDAQ_INFO("Launching remote kernel ({})", kernelName);
    std::vector<cudaq::KernelExecution> codes;
    std::string name = kernelName;
    const auto packed = args.getPacked();
    if (!packed)
      throw std::runtime_error(
          "Analog Hamiltonian launch requires a packed JSON payload.");
    std::string strArgs(reinterpret_cast<const char *>(packed->data.data()),
                        packed->data.size());
    codes.push_back(KernelExecution{.name = name, .code = strArgs});

    executor->setShots(policy.options.shots);
    return executor->execute(codes);
  }

  async_sample_result launchKernel(const async_sample_policy &policy,
                                   const CompiledModule &module,
                                   KernelArgs args) override {
    // Keep this asynchronous if requested
    return async_sample_result(
        launchKernelCommon(policy.inner, module, std::move(args)));
  }

  sample_result launchKernel(const sample_policy &policy,
                             const CompiledModule &module,
                             KernelArgs args) override {
    // Otherwise make this synchronous
    return launchKernelCommon(policy, module, std::move(args)).get();
  }
};

} // namespace cudaq
