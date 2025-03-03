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

class ArgumentConverter;

class KernelSubstitutionInfo {
public:
  KernelSubstitutionInfo(mlir::StringRef kernelName, mlir::ModuleOp substModule)
      : kernelName(kernelName), substModule(substModule) {}

  /// Some substitutions may generate global constant information. Use this
  /// interface to access both the substitutions and any global constants
  /// created.
  mlir::ModuleOp getSubstitutionModule() { return substModule; }

  /// Get the list of substitutions for this kernel that were generated
  /// by `ArgumentConverter::gen()`.
  mlir::SmallVector<cc::ArgumentSubstitutionOp> &getSubstitutions() {
    return substitutions;
  }

  mlir::StringRef getKernelName() { return kernelName; }

private:
  mlir::StringRef kernelName;
  mlir::ModuleOp substModule;
  mlir::SmallVector<cc::ArgumentSubstitutionOp> substitutions;

  friend ArgumentConverter;
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
  void gen(mlir::StringRef kernelName, mlir::ModuleOp sourceModule,
           const std::vector<void *> &arguments);

  /// Generate a substitution ModuleOp but include only the arguments that do
  /// not appear in the set of \p exclusions.
  void gen(const std::vector<void *> &arguments,
           const std::unordered_set<unsigned> &exclusions);

  /// Generate a substitution ModuleOp but drop the first \p numDrop arguments
  /// and thereby exclude them from the substitutions.
  void gen_drop_front(const std::vector<void *> &arguments, unsigned numDrop);

  /// Get the kernel info that were collected by `gen()`.
  std::list<KernelSubstitutionInfo> &getKernelSubstitutions() {
    return kernelSubstitutions;
  }

  bool isRegisteredKernel(const std::string &kernelName) {
    return std::find(nameRegistry.begin(), nameRegistry.end(), kernelName) !=
           nameRegistry.end();
  }

  std::string &registerKernel(const std::string &kernelName) {
    return nameRegistry.emplace_back(kernelName);
  }

private:
  KernelSubstitutionInfo &addKernelInfo(mlir::StringRef kernelName,
                                        mlir::ModuleOp substModule) {
    return kernelSubstitutions.emplace_back(kernelName, substModule);
  }

  /// Memory to store new kernel names generated during argument conversion.
  /// Use list here to keep references to those elements valid.
  std::list<std::string> nameRegistry;

  /// Memory to store new kernel info generated during argument conversion.
  /// Use list here to keep elements sorted in order of creation.
  std::list<KernelSubstitutionInfo> kernelSubstitutions;

  /// Original module before substitutions.
  mlir::ModuleOp sourceModule;

  /// Kernel we are substituting the arguments for.
  mlir::StringRef kernelName;
};

} // namespace cudaq::opt
