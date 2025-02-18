/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/qis/state.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include <list>
#include <unordered_set>
#include <vector>

namespace cudaq::opt {

class StateAggregator {
public:
  /// Create an instance of the state aggregator for a specified \p sourceModule.
  StateAggregator(mlir::ModuleOp moduleOp);

  /// Collect kernel names and arguments for all state arguments.
  void collect(mlir::StringRef kernelName, const std::vector<void *> &arguments);

  /// Get the list of kernel names and their arguments that were collected by `collect()`.
  std::list<std::pair<std::string, const std::vector<void *>>> &getKernelInfo() {
    return kernelInfo;
  }

private:
  void collectKernelInfo(const cudaq::state *v);

  bool hasKernelInfo(const std::string &kernelName) {
    for (auto& info: kernelInfo)
    if (info.first == kernelName)
      return true;
    return false;
  }
  
  void addKernelInfo(const std::string &kernelName, const std::vector<void *>& args) {
    kernelInfo.push_back(std::make_pair(kernelName, args));
  }

private:
  mlir::ModuleOp moduleOp;
  mlir::OpBuilder builder;

  std::list<std::pair<std::string, const std::vector<void *>>> kernelInfo;
};

} // namespace cudaq::opt
