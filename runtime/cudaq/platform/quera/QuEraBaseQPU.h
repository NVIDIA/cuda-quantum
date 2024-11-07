/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BaseRemoteRESTQPU.h"

namespace cudaq {

/// @brief The `QuEraBaseQPU` is a QPU that allows users to
// submit kernels to the QuEra machine.
class QuEraBaseQPU : public BaseRemoteRESTQPU {
protected:
  std::tuple<mlir::ModuleOp, mlir::MLIRContext *, void *>
  extractQuakeCodeAndContext(const std::string &kernelName,
                             void *data) override {
    throw std::runtime_error("Not supported on this target.");
  }

public:
  virtual bool isRemote() override { return true; }

  virtual bool isEmulated() override { return false; }

  KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    if (kernelName.find(cudaq::runtime::cudaqAHSPrefixName) != 0)
      throw std::runtime_error("Not supported on this target.");

    cudaq::info("Launching remote kernel ({})", kernelName);
    std::vector<cudaq::KernelExecution> codes;

    std::string name = kernelName;
    char *charArgs = (char *)(args);
    std::string strArgs = charArgs;
    nlohmann::json j;
    std::vector<std::size_t> mapping_reorder_idx;
    codes.emplace_back(name, strArgs, j, mapping_reorder_idx);

    cudaq::details::future future;
    future = executor->execute(codes);
    return {};
  }

  void launchKernel(const std::string &kernelName,
                    const std::vector<void *> &rawArgs) override {
    throw std::runtime_error("Not supported on this target.");
  }
};
} // namespace cudaq
