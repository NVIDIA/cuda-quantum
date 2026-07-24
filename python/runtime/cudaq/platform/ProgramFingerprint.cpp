/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ProgramFingerprint.h"
#include "common/DeviceCodeRegistry.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Target/CompileTarget.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <span>
#include <string>
#include <vector>

using namespace mlir;

namespace {
/// Feed one component into the fingerprint hasher, length-prefixed so that
/// component boundaries are unambiguous: without the prefix, the component
/// sequences ("ab", "c") and ("a", "bc") would produce the same byte stream
/// and therefore the same digest.
void appendFingerprintComponent(llvm::SHA256 &hasher,
                                llvm::StringRef component) {
  hasher.update(std::to_string(component.size()));
  hasher.update(llvm::StringRef(":"));
  hasher.update(component);
}

/// Return true if \p func is a declaration whose definition lives in the
/// C++ registered-kernel registry. Such a definition is linked in by the
/// compiler *after* fingerprinting, so the module text only ever shows the
/// declaration — two different registered implementations would fingerprint
/// identically. Kernels depending on one cannot be validated by fingerprint.
bool hasRegisteredDefinition(func::FuncOp func) {
  if (!func.isDeclaration())
    return false;
  // Recover the short kernel name the registry is keyed on: strip any
  // ".suffix" and the __nvqpp__mlirgen__ prefix.
  llvm::StringRef kernelName = func.getName();
  if (auto dot = kernelName.find('.'); dot != llvm::StringRef::npos)
    kernelName = kernelName.substr(0, dot);
  if (kernelName.starts_with(cudaq::runtime::cudaqGenPrefixName))
    kernelName = kernelName.substr(cudaq::runtime::cudaqGenPrefixLength);
  return !cudaq::get_quake_by_name(kernelName.str(),
                                   /*throwException=*/false)
              .empty();
}
} // namespace

/// Compute a deterministic fingerprint of the exact program that compilation
/// would see: the module with every callable closure merged in, plus every
/// argument substitution generated for compile-time-bound (callable)
/// arguments. Two launches whose fingerprints match are guaranteed to compile
/// to the same artifact, so the digest validates reuse of the cached module.
///
/// Returns `std::nullopt` when the program has a dependency whose
/// implementation is not owned by the module — a declaration backed by the
/// C++ registered-kernel registry, or a `cc.device_call` — because the module
/// text cannot vouch for code that lives outside it. Callers must then treat
/// the launch as non-cacheable (compile every call).
///
/// On return, \p resolvedModule holds the merged clone. On a cache miss the
/// caller can compile it directly.
std::optional<std::array<std::uint8_t, 32>>
cudaq::detail::createProgramFingerprint(
    const std::string &name, mlir::ModuleOp mod,
    const std::vector<void *> &rawArgs, const cudaq::CompileTarget &target,
    mlir::OwningOpRef<mlir::ModuleOp> &resolvedModule) {
  resolvedModule = mod.clone();
  cudaq_internal::compiler::mergeAllCallableClosures(resolvedModule.get(), name,
                                                     rawArgs);

  // Refuse to fingerprint programs with unowned dependencies.
  bool hasUnownedDependency = false;
  resolvedModule->walk([&](func::FuncOp func) {
    if (hasRegisteredDefinition(func))
      hasUnownedDependency = true;
  });
  resolvedModule->walk(
      [&](cudaq::cc::DeviceCallOp) { hasUnownedDependency = true; });
  if (hasUnownedDependency)
    return std::nullopt;

  // Digest component 1: the printed IR of the resolved module. The MLIR
  // printer assigns SSA names deterministically from structure, so two
  // structurally identical modules print byte-identically.
  llvm::SHA256 hasher;
  {
    std::string moduleText;
    llvm::raw_string_ostream moduleStream(moduleText);
    resolvedModule->print(moduleStream);
    appendFingerprintComponent(hasher, moduleText);
  }

  // No arguments — nothing gets synthesized into the program.
  if (rawArgs.empty())
    return hasher.final();

  // Digest components 2..n: the argument substitutions that compilation will
  // bake into the artifact.
  auto entryPoint = cudaq::getKernelFuncOp(resolvedModule.get(), name);
  std::span<void *const> synthesisArgs{rawArgs};
  // Must outlive `gen()`: `retainCallableArguments` re-points `synthesisArgs`
  // at this vector.
  std::vector<void *> closureArgs;
  if (!cudaq::opt::factory::isFullySynthesized(entryPoint))
    cudaq_internal::compiler::retainCallableArguments(synthesisArgs,
                                                      closureArgs, entryPoint);

  cudaq_internal::compiler::ArgumentConverter argConverter(
      name, resolvedModule.get(), target.isLocalSimulator);
  argConverter.gen(synthesisArgs);
  for (auto *substitution : argConverter.getKernelSubstitutions()) {
    appendFingerprintComponent(hasher, substitution->getKernelName());
    auto substitutionModule = substitution->getSubstitutionModule();
    std::string substitutionText;
    llvm::raw_string_ostream substitutionStream(substitutionText);
    substitutionModule.print(substitutionStream);
    appendFingerprintComponent(hasher, substitutionText);
    // The substitution modules are parentless top-level ops created by
    // `gen()`; `~ArgumentConverter` frees the info structs but not the
    // modules. This runs on every launch, so erase them here or leak IR
    // per call.
    substitutionModule.erase();
  }
  return hasher.final();
}
