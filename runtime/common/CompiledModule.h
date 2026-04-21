/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/Resources.h"
#include "common/ThunkInterface.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

// This header file and the types defined within are designed to have no
// dependencies and be useable across the compiler and runtime. Constructing
// `CompiledModule` is supported through
// `cudaq_internal::compiler::CompiledModuleHelper`, available in
// `CompiledModuleHelper.h` from `cudaq-mlir-runtime`.

namespace mlir {
class ExecutionEngine;
class MLIRContext;
} // namespace mlir

namespace cudaq_internal::compiler {
class CompiledModuleHelper;
} // namespace cudaq_internal::compiler

namespace cudaq {

/// JitEngine is a type-erased class that is wrapping an mlir::ExecutionEngine
/// without introducing any link time dependency on MLIR for the client of the
/// class. Memory management for of the mlir::ExecutionEngine is handled
/// internally.
class JitEngine {
  using RawFnPtr = void (*)();
  using LookupFn = std::function<RawFnPtr(const std::string &)>;

  struct Base {
    LookupFn lookupFn;
    std::function<void(const std::string &)> runFn;
  };

public:
  JitEngine(std::unique_ptr<mlir::ExecutionEngine>);

  void run(const std::string &kernelName) const { impl->runFn(kernelName); }

  void (*lookupRawNameOrFail(const std::string &kernelName) const)() {
    return impl->lookupFn(kernelName);
  }

  std::size_t getKey() const;

private:
  class Impl;
  std::shared_ptr<Base> impl;
};

/// Pre-computed result metadata, set at build time. Used at execution time
/// for result buffer allocation and type conversion.
class ResultInfo {
  friend class cudaq_internal::compiler::CompiledModuleHelper;
  friend class CompiledModule;

  /// Opaque pointer to the `mlir::Type` of the result. Obtained via
  /// `mlir::Type::getAsOpaquePointer()`.
  /// Lifetime: the `MLIRContext` that owns the Type must outlive this object.
  const void *typeOpaquePtr = nullptr;

  /// Size (in bytes) of the buffer needed to hold the result value.
  /// Pre-computed from the MLIR type at build time.
  std::size_t bufferSize = 0;

  /// Pre-computed struct field offsets (from `getTargetLayout`). Only non-empty
  /// for struct return types.
  std::vector<std::size_t> fieldOffsets;

public:
  /// Whether this kernel has a result that must be marshaled.
  bool hasResult() const { return typeOpaquePtr != nullptr; }
};

/// @brief A compiled MLIR module, ready for execution or code generation.
///
/// Contains any number of named compilation artifacts (we currently support
/// JIT binaries and optimized MLIR modules) that result from the compilation
/// of a Quake MLIR module.
///
/// This type does not depend on MLIR/LLVM — it only keeps type-erased / opaque
/// pointers. Build instances with
/// `cudaq_internal::compiler::CompiledModuleHelper`.
class CompiledModule {
public:
  // --- Compiled artifact types ---

  /// JIT-compiled artifact, ready for local execution.
  class JitArtifact {
    JitEngine engine;
    void (*entryPoint)() = nullptr;
    std::int64_t (*argsCreator)(const void *, void **) = nullptr;
    /// Offset (in bytes) of the result field within the argsCreator-packed
    /// buffer. Only valid when argsCreator is non-null and the kernel has a
    /// result. Use resultInfo.bufferSize to know how many bytes to copy.
    std::int64_t (*returnOffset)() = nullptr;
    std::optional<Resources> resourceCounts;

    JitArtifact(JitEngine engine, void (*entryPoint)(),
                int64_t (*argsCreator)(const void *, void **),
                int64_t (*returnOffset)(),
                std::optional<Resources> resourceCounts)
        : engine(engine), entryPoint(entryPoint), argsCreator(argsCreator),
          returnOffset(returnOffset),
          resourceCounts(std::move(resourceCounts)) {}

    friend class CompiledModule;
    friend class cudaq_internal::compiler::CompiledModuleHelper;

  public:
    // TODO: remove the following two methods once the `CompiledModule` instance
    // is returned to Python.

    /// @brief Get the entry point of the kernel as a function pointer.
    ///
    /// Assumes that there is (exactly one) compiled JIT artifact.
    ///
    /// The returned function pointer will expect different arguments depending
    /// on the kernel:
    ///  - if the kernel returns a value and/or is not fully specialized, the
    ///    entry point will expect a pointer to a buffer storing the packed
    ///    arguments and result.
    ///  - otherwise, the entry point will not expect any arguments.
    ///
    /// Prefer using `CompiledModule::execute` instead of calling this function
    /// as it will handle the buffer and argument packing automatically.
    void (*getEntryPoint() const)();
    JitEngine getEngine() const;

    std::optional<Resources> getResourceCounts() const;
  };

  /// Optimized MLIR module artifact, for deferred code generation or
  /// re-targeting.
  /// Type-erased to keep this header MLIR-free.
  class MlirArtifact {
    /// Opaque ModuleOp pointer (via `module.getAsOpaquePointer()`).
    const void *modulePtr = nullptr;

    /// Optional owning reference to the containing `MLIRContext`.
    std::shared_ptr<mlir::MLIRContext> context;

    MlirArtifact(const void *modulePtr,
                 std::shared_ptr<mlir::MLIRContext> context)
        : modulePtr(modulePtr), context(std::move(context)) {}

    friend class CompiledModule;
    friend class cudaq_internal::compiler::CompiledModuleHelper;
  };

  /// A compiled artifact is either a JIT binary or an MLIR module.
  using CompiledArtifact = std::variant<JitArtifact, MlirArtifact>;

  // --- Compilation metadata ---

  /// Metadata on the compilation artifacts.
  struct CompilationMetadata {
    /// Qubit reorder indices emitted by the qubit-mapping pass.
    std::vector<std::size_t> reorderIdx;
  };

  // --- Queries ---

  /// Whether any artifact in the map is a JitArtifact.
  bool hasJit() const;

  /// Whether any artifact in the map is an MlirArtifact.
  bool hasMlir() const;

  /// Get the compiled JIT artifact. Returns the first one found.
  ///
  /// Throws if none exists.
  const JitArtifact &getJit() const;

  /// Get the optimized MLIR artifact. Returns the first one found.
  ///
  /// Throws if none exists.
  const MlirArtifact &getMlir() const;

  /// Get all compiled artifacts.
  const std::map<std::string, CompiledArtifact> &getArtifacts() const {
    return artifacts;
  }

  /// Whether the kernel is fully specialized (all arguments inlined). For JIT
  /// kernels this means `argsCreator` is null.
  /// Kernels without a JIT artifact are considered fully specialized.
  bool isFullySpecialized() const;

  const std::string &getName() const { return name; }
  const ResultInfo &getResultInfo() const { return resultInfo; }
  const CompilationMetadata &getMetadata() const { return metadata; }

  // --- Execution (local JIT path) ---

  /// @brief Execute a fully specialized kernel (no external arguments needed).
  ///
  /// Assumes that there is (exactly one) compiled JIT artifact.
  KernelThunkResultType execute() const;

  /// @brief Execute the JIT-ed kernel with caller-provided arguments.
  ///
  /// Assumes that there is (exactly one) compiled JIT artifact.
  KernelThunkResultType execute(const std::vector<void *> &rawArgs) const;

private:
  friend class cudaq_internal::compiler::CompiledModuleHelper;

  CompiledModule(std::string kernelName);

  /// Add a compiled artifact to the module under the given name.
  void addArtifact(std::string name, CompiledArtifact artifact);

  std::string name;
  ResultInfo resultInfo; // TODO: we might want to store the entire kernel
                         // signature here. Though I'm not sure what MLIR
                         // agnostic information is worth storing.
  CompilationMetadata metadata;
  std::map<std::string, CompiledArtifact> artifacts;
};

} // namespace cudaq
