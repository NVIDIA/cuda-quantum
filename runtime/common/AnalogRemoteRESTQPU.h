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

/// @brief Base QPU class for analog platforms like `quera` and `pasqal`.
/// Provides common functionality and implementation.
class AnalogRemoteRESTQPU : public BaseRemoteRESTQPU {
public:
  /// @brief Check if this is a remote target
  virtual bool isRemote() override { return true; }

  /// @brief Check if this is an emulated target
  virtual bool isEmulated() override { return false; }

  /// @brief Launch a kernel with the given arguments
  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    throw std::runtime_error(
        "Arbitrary kernel execution is not supported on this target.");
  }

  /// @brief Launch a kernel with the given arguments
  /// Only analog Hamiltonian kernels are supported
  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    if (kernelName.find(cudaq::runtime::cudaqAHKPrefixName) != 0)
      throw std::runtime_error(
          "Arbitrary kernel execution is not supported on this target.");

    if (emulate)
      throw std::runtime_error(
          "Local emulation is not yet supported on this target.");

    CUDAQ_INFO("Launching remote kernel ({})", kernelName);
    std::vector<cudaq::KernelExecution> codes;
    std::string name = kernelName;
    char *charArgs = (char *)(args);
    std::string strArgs = charArgs;
    nlohmann::json j;
    std::vector<std::size_t> mapping_reorder_idx;
    codes.emplace_back(name, strArgs, j, mapping_reorder_idx);

    if (executionContext) {
      executor->setShots(executionContext->shots);
      cudaq::details::future future;
      future = executor->execute(codes);
      // Keep this asynchronous if requested
      if (executionContext->asyncExec) {
        executionContext->asyncResult = async_sample_result(std::move(future));
        return {};
      }
      // Otherwise make this synchronous
      executionContext->result = future.get();
    }
    return {};
  }
};

} // namespace cudaq
