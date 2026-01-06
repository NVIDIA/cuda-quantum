/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
#include <unordered_set>
#include <vector>

namespace cudaq {
namespace opt {

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

  ~ArgumentConverter() {
    for (auto *kInfo : kernelSubstitutions) {
      delete kInfo;
    }
  }

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
  mlir::SmallVector<KernelSubstitutionInfo *> &getKernelSubstitutions() {
    return kernelSubstitutions;
  }

  bool isRegisteredKernel(mlir::StringRef kernelName) {
    return std::find(nameRegistry.begin(), nameRegistry.end(),
                     kernelName.str()) != nameRegistry.end();
  }

  mlir::StringRef registerKernel(mlir::StringRef kernelName) {
    return nameRegistry.emplace_back(
        mlir::StringAttr::get(sourceModule.getContext(), kernelName));
  }

private:
  KernelSubstitutionInfo *addKernelInfo(mlir::StringRef kernelName,
                                        mlir::ModuleOp substModule) {
    return kernelSubstitutions.emplace_back(
        new KernelSubstitutionInfo(kernelName, substModule));
  }

  /// Memory to store new kernel names generated during argument conversion.
  mlir::SmallVector<mlir::StringAttr> nameRegistry;

  /// Memory to store new kernel info generated during argument conversion.
  mlir::SmallVector<KernelSubstitutionInfo *> kernelSubstitutions;

  /// Original module before substitutions.
  mlir::ModuleOp sourceModule;

  /// Kernel we are substituting the arguments for.
  mlir::StringRef kernelName;
};
} // namespace opt

namespace detail {
/// Merge modules from any CallableClosureArgument arguments into \p intoModule.
/// The \p rawArgs must correspond to the entry point function with the short
/// name \p shortName that must appear in \p intoModule.
///
/// This merging step is done explicitly because: (1) it can be under better
/// control as MLIR objects and (2) doing so eliminates the round-trip cost of
/// dumping entire (filtered?) modules to strings and then converting them back
/// to binary form.
///
/// Return <code>true</code> if and only if \p intoModule has been modified.
bool mergeAllCallableClosures(mlir::ModuleOp intoModule,
                              const std::string &shortName,
                              const std::vector<void *> &rawArgs,
                              std::optional<unsigned> betaRedux = {});
} // namespace detail

} // namespace cudaq
