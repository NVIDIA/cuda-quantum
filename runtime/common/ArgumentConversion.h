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

  
class KernelInfo {
  public:
    KernelInfo(mlir::OpBuilder builder, mlir::StringRef kernelName)
    :  kernelName(kernelName) {
      substModule = builder.create<mlir::ModuleOp>(builder.getUnknownLoc());
    }
  
    /// Some substitutions may generate global constant information. Use this
    /// interface to access both the substitutions and any global constants
    /// created.
    mlir::ModuleOp getSubstitutionModule() {
      return substModule;
    }

    /// Get the list of substitutions for this kernel that were generated
    /// by `ArgumentConverter::gen()`.
    mlir::SmallVector<cc::ArgumentSubstitutionOp> &getSubstitutions() {
      return substitutions;
    }

  private:
    mlir::ModuleOp substModule;
    mlir::StringRef kernelName;
    mlir::SmallVector<cc::ArgumentSubstitutionOp> substitutions;
  };

  
class ArgumentConverter {
public:
  /// Build an instance to create argument substitutions for a specified \p
  /// kernelName in \p sourceModule.
  ArgumentConverter(mlir::StringRef kernelName, mlir::ModuleOp sourceModule);

  /// Generate a substitution ModuleOp for the vector of arguments presented.
  /// The arguments are those presented to the kernel, kernelName.
  void gen(const std::vector<void *> &arguments);

  /// Generate a substitution ModuleOp for the vector of arguments presented.
  /// The arguments are those presented to the kernel, kernelName.
  void gen(mlir::StringRef kernelName, const std::vector<void *> &arguments);

  /// Generate a substitution ModuleOp but include only the arguments that do
  /// not appear in the set of \p exclusions.
  void gen(const std::vector<void *> &arguments,
           const std::unordered_set<unsigned> &exclusions);

  /// Generate a substitution ModuleOp but drop the first \p numDrop arguments
  /// and thereby exclude them from the substitutions.
  void gen_drop_front(const std::vector<void *> &arguments, unsigned numDrop);

  /// Kernel we are converting the arguments for.
  mlir::StringRef getKernelName() { return kernelName; }

  /// Get the map of kernel names to their kernel info that
  /// were collected by `collect()`.
   mlir::DenseMap<mlir::StringRef, KernelInfo>& getKernelInfo() {
    return kernelInfo;
  }

  bool isRegisteredKernel(const std::string &kernelName) {
    return std::find(nameRegistry.begin(), nameRegistry.end(), kernelName) != nameRegistry.end();
  }

  std::string &registerKernel(const std::string &kernelName) {
    return nameRegistry.emplace_back(kernelName);
  }

  KernelInfo& addKernelInfo(mlir::StringRef kernelName) {
    auto [it,b] = kernelInfo.try_emplace(kernelName, std::move(KernelInfo(builder, kernelName)));
    return it->second;
  }

  private:
  /// Memory to store new kernel names generated during argument conversion.
  std::list<std::string> nameRegistry;

  /// Kernel info for kernels we are converting the arguments for, including
  /// new kernels generated from state arguments.
  mlir::DenseMap<mlir::StringRef, KernelInfo> kernelInfo;

  mlir::ModuleOp sourceModule;
  mlir::OpBuilder builder;
  mlir::StringRef kernelName;
};

} // namespace cudaq::opt
