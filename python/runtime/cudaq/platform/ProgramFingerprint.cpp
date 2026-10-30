/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ProgramFingerprint.h"
#include "common/ArgumentWrapper.h"
#include "common/DeviceCodeRegistry.h"
#include "common/Environment.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "utils/OpaqueArguments.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Target/CompileTarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include <cstdint>
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

/// Alternative fingerprinting method that hashes the structure of the program.
/// Enabled by `CUDAQ_FINGERPRINT_STRUCTURE`.
std::optional<std::array<std::uint8_t, 32>>
fingerprintByStructure(const std::string &name, mlir::ModuleOp mod,
                       const std::vector<void *> &rawArgs);
bool useStructuralFingerprint();

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
    const std::vector<void *> &rawArgs,
    mlir::OwningOpRef<mlir::ModuleOp> &resolvedModule) {
  if (useStructuralFingerprint())
    return fingerprintByStructure(name, mod, rawArgs);

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
  // Note that it is safe to ignore any differences in arguments synthesis due
  // to compile options/target, as those are hashed separately anyways.
  auto entryPoint = cudaq::getKernelFuncOp(resolvedModule.get(), name);
  std::span<void *const> synthesisArgs{rawArgs};
  // Must outlive `gen()`: `retainCallableArguments` re-points `synthesisArgs`
  // at this vector.
  std::vector<void *> closureArgs;
  if (!cudaq::opt::factory::isFullySynthesized(entryPoint))
    cudaq_internal::compiler::retainCallableArguments(synthesisArgs,
                                                      closureArgs, entryPoint);

  cudaq_internal::compiler::ArgumentConverter argConverter(
      name, resolvedModule.get());
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

namespace {
/// Summarize a program's structure for the fingerprint as opposed to printing
/// the module to text with a pre-order walk over every operation under the
/// root, hashing enough per operation to reconstruct the IR graph up to naming.
///
/// Attributes and types are mixed in by uniqued pointer rather than by content.
/// Within one MLIRContext the context's uniquer guarantees pointer equality if
/// and only if content equality, and it owns them for its whole lifetime, so
/// an address cannot be recycled under a different value.
///
/// Operations carry no such guarantee so values and blocks are mixed in by
/// position in the traversal instead of by address.
class ProgramStructureHasher {
public:
  /// Mix in every operation under \p root, in traversal order.
  void hash(mlir::Operation *root) {
    root->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
      mix(op->getName().getStringRef());
      mixPointer(op->getRawDictionaryAttrs().getAsOpaquePointer());
      // Inherent attributes live in properties rather than in the attribute
      // dictionary, so an operation's defining values — a constant's value, for
      // one — are invisible without this.
      mixHash(op->hashProperties());
      mix(op->getNumResults());
      for (auto type : op->getResultTypes())
        mixPointer(type.getAsOpaquePointer());
      // Number results before operands are mixed in, so a value used by a
      // later operation is already numbered when that use is hashed.
      for (auto result : op->getResults())
        number(result.getAsOpaquePointer());
      mix(op->getNumOperands());
      for (auto operand : op->getOperands())
        mix(number(operand.getAsOpaquePointer()));
      mix(op->getNumRegions());
      for (auto &region : op->getRegions()) {
        mix(region.getBlocks().size());
        for (auto &block : region) {
          mix(number(&block));
          mix(block.getNumArguments());
          for (auto argument : block.getArguments()) {
            mixPointer(argument.getType().getAsOpaquePointer());
            number(argument.getAsOpaquePointer());
          }
        }
      }
      // Successors are edges between blocks, so they are numbered as blocks.
      mix(op->getNumSuccessors());
      for (auto *successor : op->getSuccessors())
        mix(number(successor));
    });
  }

  /// The accumulated hash, as bytes for the fingerprint hasher.
  std::string bytes() const {
    auto value = static_cast<std::uint64_t>(digest);
    std::string out(sizeof(value), '\0');
    for (unsigned i = 0; i < sizeof(value); ++i)
      out[i] = static_cast<char>((value >> (8 * i)) & 0xff);
    return out;
  }

private:
  void mix(llvm::StringRef text) {
    digest = llvm::hash_combine(digest, llvm::hash_value(text));
  }

  void mix(std::size_t value) { digest = llvm::hash_combine(digest, value); }

  void mixHash(llvm::hash_code value) {
    digest = llvm::hash_combine(digest, value);
  }

  void mixPointer(const void *pointer) {
    digest = llvm::hash_combine(digest, llvm::hash_value(pointer));
  }

  /// Position of \p handle in traversal order, assigning one if it is new.
  ///
  /// Values and blocks share the numbering: they are distinct live objects, so
  /// their addresses cannot collide, and a single sequence keeps the assignment
  /// order easy to reason about.
  std::size_t number(const void *handle) {
    return numbering.try_emplace(handle, numbering.size()).first->second;
  }

  llvm::hash_code digest = llvm::hash_value(0);
  llvm::DenseMap<const void *, std::size_t> numbering;
};

/// Structural digest of \p mod, as a fingerprint component.
std::string programStructure(mlir::Operation *mod) {
  ProgramStructureHasher hasher;
  hasher.hash(mod);
  return hasher.bytes();
}

/// The closure modules that resolving a launch's callable arguments would add
/// to the program, in the order `mergeAllCallableClosures` would add them.
class ResolvedProgram {
public:
  ResolvedProgram(ModuleOp mod) : mod(mod) {}

  /// Collect the closures reachable from \p rawArgs bound to \p entryPoint.
  /// \p offset is the merge's beta-reduction position: the number of leading
  /// parameters that \p rawArgs does not describe.
  void resolve(func::FuncOp entryPoint, std::span<void *const> rawArgs,
               unsigned offset = 0) {
    if (rawArgs.empty())
      return;
    auto inputs = entryPoint.getFunctionType().getInputs();
    for (auto [i, type] : llvm::enumerate(inputs.drop_front(offset))) {
      if (!isa<cudaq::cc::CallableType>(type) || i >= rawArgs.size() ||
          !rawArgs[i])
        continue;
      auto &closure =
          *reinterpret_cast<cudaq::runtime::CallableClosureArgument *>(
              rawArgs[i]);
      auto name = cudaq::runtime::cudaqGenPrefixName + closure.getShortName();
      if (isDefined(name))
        continue;
      auto closureModule = closure.getModule();
      closureModules.push_back(closureModule);
      closureModule.walk([&](func::FuncOp func) {
        if (!func.isDeclaration())
          resolvedNames.insert(func.getName());
      });
      if (auto closureEntry = closureModule.lookupSymbol<func::FuncOp>(name))
        resolve(closureEntry, closure.getArgs(),
                closure.getStartLiftedPos().value_or(0));
    }
  }

  llvm::ArrayRef<ModuleOp> getClosureModules() const { return closureModules; }

  /// True if the resolved program does not own \p func's definition: a
  /// declaration backed by the C++ registry that no closure supplies.
  bool isUnowned(func::FuncOp func) const {
    return hasRegisteredDefinition(func) &&
           !resolvedNames.contains(func.getName());
  }

private:
  /// True if the program already holds a definition of \p name, in which case
  /// the merge leaves a closure of that name out.
  bool isDefined(llvm::StringRef name) {
    if (resolvedNames.contains(name))
      return true;
    auto func = mod.lookupSymbol<func::FuncOp>(name);
    return func && !func.isDeclaration();
  }

  ModuleOp mod;
  llvm::SmallVector<ModuleOp> closureModules;
  /// Names the closures define, so that a declaration of one is not mistaken
  /// for an unowned dependency and a second closure of the same name is not
  /// counted twice.
  llvm::StringSet<> resolvedNames;
};

/// Fingerprint by describing the resolved program rather than building it.
/// \p mod is left untouched, so the caller gets no clone to compile.
std::optional<std::array<std::uint8_t, 32>>
fingerprintByStructure(const std::string &name, mlir::ModuleOp mod,
                       const std::vector<void *> &rawArgs) {
  auto entryPoint = cudaq::getKernelFuncOp(mod, name);
  ResolvedProgram program(mod);
  program.resolve(entryPoint, rawArgs);

  // Refuse to fingerprint programs with unowned dependencies.
  bool hasUnownedDependency = false;
  auto checkOwnership = [&](ModuleOp module) {
    module.walk([&](func::FuncOp func) {
      if (program.isUnowned(func))
        hasUnownedDependency = true;
    });
    module.walk([&](cudaq::cc::DeviceCallOp) { hasUnownedDependency = true; });
  };
  checkOwnership(mod);
  for (auto closureModule : program.getClosureModules())
    checkOwnership(closureModule);
  if (hasUnownedDependency)
    return std::nullopt;

  // Digest components 1..k: the structure of the module and of each closure
  // that resolving it adds, each walked in a fixed order so that two
  // structurally identical programs summarize identically.
  llvm::SHA256 hasher;
  appendFingerprintComponent(hasher, programStructure(mod));
  for (auto closureModule : program.getClosureModules())
    appendFingerprintComponent(hasher, programStructure(closureModule));

  // No arguments — nothing gets synthesized into the program.
  if (rawArgs.empty())
    return hasher.final();

  // Remaining digest components: the argument substitutions that compilation
  // will bake into the artifact. These are what tie a closure to the argument
  // it is bound to, and they carry any global constants a binding generates.
  // Note that it is safe to ignore any differences in arguments synthesis due
  // to compile options/target, as those are hashed separately anyways.
  //
  // The converter reads the entry point and the data layout from the module it
  // is given, and finds a callable's callee in that closure's own module, so
  // the unmerged module serves it as well as a merged one would.
  std::span<void *const> synthesisArgs{rawArgs};
  // Must outlive `gen()`: `retainCallableArguments` re-points `synthesisArgs`
  // at this vector.
  std::vector<void *> closureArgs;
  if (!cudaq::opt::factory::isFullySynthesized(entryPoint))
    cudaq_internal::compiler::retainCallableArguments(synthesisArgs,
                                                      closureArgs, entryPoint);

  cudaq_internal::compiler::ArgumentConverter argConverter(name, mod);
  argConverter.gen(synthesisArgs);
  for (auto *substitution : argConverter.getKernelSubstitutions()) {
    appendFingerprintComponent(hasher, substitution->getKernelName());
    auto substitutionModule = substitution->getSubstitutionModule();
    appendFingerprintComponent(hasher, programStructure(substitutionModule));
    // The substitution modules are parentless top-level ops created by
    // `gen()`; `~ArgumentConverter` frees the info structs but not the
    // modules. This runs on every launch, so erase them here or leak IR
    // per call.
    substitutionModule.erase();
  }
  return hasher.final();
}

/// True when `CUDAQ_FINGERPRINT_STRUCTURE` selects the structural digest over
/// printing the resolved module. Read once: this is on every launch.
bool useStructuralFingerprint() {
  static const bool enabled =
      cudaq::getEnvBool("CUDAQ_FINGERPRINT_STRUCTURE", false);
  return enabled;
}
} // namespace
