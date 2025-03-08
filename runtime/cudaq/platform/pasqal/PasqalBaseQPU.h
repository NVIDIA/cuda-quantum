/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The `PasqalBaseQPU` is a QPU that allows users to submit kernels to
/// the Pasqal machine.
class PasqalBaseQPU : public BaseRemoteRESTQPU {
protected:
  virtual std::tuple<mlir::ModuleOp, mlir::MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {

    throw std::runtime_error("Not supported on this target.");
  }

public:
  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    throw std::runtime_error(
        "Arbitrary kernel execution is not supported on this target.");
  }

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

    cudaq::info("Launching remote kernel ({})", kernelName);
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
