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
 #include "ArgumentConversion.h"
 
 namespace cudaq::opt {
  struct KernelInfo {
    ArgumentConverter converter;
    const std::vector<void *> args;
  };

 class StateAggregator {
 public:
   /// Create an instance of the state aggregator for a specified \p
   /// sourceModule.
   StateAggregator(){}
 
   /// Collect kernel names and arguments for all state arguments.
   void collect(mlir::ModuleOp moduleOp, const std::string& kernelName,
                const std::vector<void *> &arguments);
 
   /// Get the map of kernel names to their kernel info that
   /// were collected by `collect()`.
   std::list<KernelInfo>& getKernelInfo() {
     return kernelInfo;
   }
 
 private:
   void collectKernelInfo(mlir::ModuleOp moduleOp, const cudaq::state *v);
 
   bool hasKernelInfo(const std::string &kernelName) {
     return std::find(nameRegistry.begin(), nameRegistry.end(), kernelName) != nameRegistry.end();
   }
 
   KernelInfo& addKernelInfo(mlir::ModuleOp moduleOp, const std::string &kernelName,
                      const std::vector<void *> &args) {
    auto &name = nameRegistry.emplace_back(kernelName);
    return kernelInfo.emplace_back(std::move(ArgumentConverter(name, moduleOp)), args);
   }
 
 private:
   /// Memory to store new kernel names generated during argument conversion.
   std::list<std::string> nameRegistry;

   /// Kernel info for kernels we are converting the arguments for, including
   /// new kernels generated from state arguments.
   std::list<KernelInfo> kernelInfo;
 };
 
 } // namespace cudaq::opt