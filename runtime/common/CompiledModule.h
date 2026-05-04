/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/NamedVariantStore.h"
#include "common/Resources.h"
#include "common/ThunkInterface.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
  /// Get the size (in bytes) of the buffer needed to hold the result value.
  std::size_t getBufferSize() const { return bufferSize; }
};

/// @brief A bundle of artifacts that store the various representations of a
/// Quake kernel.
///
/// Contains any number of named artifacts (we currently support JIT binaries,
/// optimized MLIR modules, and pre-computed resource metrics) that result from
/// the definition, or processing of a Quake MLIR module.
///
/// This type does not depend on MLIR/LLVM — it only keeps type-erased / opaque
/// pointers. Build instances with
/// `cudaq_internal::compiler::CompiledModuleHelper`.
class FatQuakeModule {
public:
  // --- Compiled artifact types ---

  /// JIT-compiled artifact, ready for local execution.
  class JitArtifact {
    JitEngine engine;
    void (*fn)() = nullptr;

    JitArtifact(JitEngine engine, void (*fn)())
        : engine(std::move(engine)), fn(fn) {}

    friend class FatQuakeModule;
    friend class cudaq_internal::compiler::CompiledModuleHelper;

  public:
    /// Get the raw function pointer stored in this artifact.
    void (*getFn() const)();
    JitEngine getEngine() const;
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

    friend class FatQuakeModule;
    friend class SourceModule;
    friend class cudaq_internal::compiler::CompiledModuleHelper;

  public:
    /// Get the opaque pointer to the MLIR module.
    const void *getOpaqueModulePtr() const { return modulePtr; }
  };

  /// Pre-compiled kernel, stored as a raw function pointer.
  ///
  /// For kernels whose thunk is already present in the pre-compiled binary
  /// generated by `nvq++`.
  class FunctionPtrArtifact {
    KernelThunkType fn = nullptr;

  public:
    explicit FunctionPtrArtifact(KernelThunkType fn) : fn(fn) {}

    /// Get the raw function pointer stored in this artifact.
    KernelThunkType getFn() const { return fn; }
  };

  /// Pre-computed resource metrics (gate counts, depth) from IR analysis.
  class ResourcesArtifact {
    Resources resources;

    ResourcesArtifact(Resources resources) : resources(std::move(resources)) {}

    friend class FatQuakeModule;
    friend class cudaq_internal::compiler::CompiledModuleHelper;

  public:
    const Resources &getResources() const { return resources; }
  };

  /// A compiled artifact is a JIT binary, an MLIR module, a raw kernel entry
  /// point, or resource metrics.
  using ArtifactsStore =
      detail::NamedVariantStore<JitArtifact, MlirArtifact, ResourcesArtifact,
                                FunctionPtrArtifact>;
  using CompiledArtifact = ArtifactsStore::Value;

  // --- Compilation metadata ---

  /// Metadata on the compilation artifacts.
  struct CompilationMetadata {
    /// Qubit reorder indices emitted by the qubit-mapping pass.
    std::vector<std::size_t> reorderIdx;
  };

  // --- Queries ---

  /// Get the JIT artifact with the given name.
  ///
  /// If no name is provided, defaults to the kernel name.
  std::optional<JitArtifact> getJit() const;
  std::optional<JitArtifact> getJit(std::string_view jitName) const;

  /// Get the MLIR artifact with the given name.
  ///
  /// If no name is provided, defaults to the kernel name.
  std::optional<MlirArtifact> getMlir() const;
  std::optional<MlirArtifact> getMlir(std::string_view mlirName) const;

  /// Get the raw function pointer artifact with the given name.
  ///
  /// If no name is provided, defaults to the kernel name.
  std::optional<FunctionPtrArtifact> getFunctionPtr() const;
  std::optional<FunctionPtrArtifact>
  getFunctionPtr(std::string_view fnName) const;

  /// Get the pre-computed resource counts, or `nullptr` if it does not exist.
  ///
  /// If no name is provided, defaults to the kernel name.
  const Resources *getResources() const;
  const Resources *getResources(std::string_view resourcesName) const;

  /// Get all compiled artifacts in insertion order.
  const ArtifactsStore &getArtifacts() const { return artifacts; }

  /// Get all MLIR artifacts in insertion order.
  auto getMlirArtifacts() const {
    return artifacts.getAllOfType<MlirArtifact>();
  }

  /// Whether the kernel is fully specialized (all arguments inlined).
  ///
  /// Currently, kernels are considered fully specialized if and only if they do
  /// not have an `argsCreator` artifact.
  bool isFullySpecialized() const;

  /// Get the argument-marshaling function, or `nullptr` if it does not exist.
  ///
  /// Assumes the artifact is named `kernelName + ".argsCreator"`.
  int64_t (*getArgsCreator() const)(const void *, void **);

  /// Get the offset (in bytes) of the result field within the
  /// `argsCreator`-packed buffer, evaluating the stored JIT function.
  /// Returns `std::nullopt` if no `.returnOffset` artifact was emitted
  /// (e.g. the kernel has no result or is fully specialized).
  ///
  /// Assumes the artifact is named `kernelName + ".returnOffset"`.
  std::optional<std::int64_t> getReturnOffset() const;

  const std::string &getName() const { return name; }
  const ResultInfo &getResultInfo() const { return resultInfo; }
  const CompilationMetadata &getMetadata() const { return metadata; }

protected:
  FatQuakeModule(std::string kernelName);

  /// Add a compiled artifact to the module under the given name.
  void addArtifact(std::string name, CompiledArtifact artifact);

  std::string name;
  ResultInfo resultInfo; // TODO: we might want to store the entire kernel
                         // signature here. Though I'm not sure what MLIR
                         // agnostic information is worth storing.
  CompilationMetadata metadata;
  ArtifactsStore artifacts;
};

/// @brief A compiled MLIR module, ready for execution or code generation.
///
/// Contains any number of named compilation artifacts (we currently support
/// JIT binaries, optimized MLIR modules, and pre-computed resource metrics)
/// that result from the compilation of a Quake MLIR module.
///
/// This type does not depend on MLIR/LLVM — it only keeps type-erased / opaque
/// pointers. Build instances with
/// `cudaq_internal::compiler::CompiledModuleHelper`.
class CompiledModule : public FatQuakeModule {
private:
  friend class cudaq_internal::compiler::CompiledModuleHelper;

  CompiledModule(std::string kernelName)
      : FatQuakeModule(std::move(kernelName)) {}
};

/// Bundle of artifacts that define a CUDA-Q kernel to be compiled and executed.
///
/// Contains either a `nvq++`-compiled function pointer or an MLIR module,
/// depending on the provenance of the kernel.
class SourceModule : public FatQuakeModule {
public:
  SourceModule(std::string kernelName)
      : FatQuakeModule(std::move(kernelName)) {}

  /// Construct a module defined by a raw kernel thunk (C++).
  SourceModule(std::string kernelName, KernelThunkType fn);

  /// Construct a module defined by an MLIR module (Python).
  ///
  /// The \p mlirModuleOpaquePtr must be a valid pointer obtained via
  /// mlir::ModuleOp::getAsOpaquePointer() and its lifetime must outlive
  /// the `SourceModule` instance.
  SourceModule(std::string kernelName, const void *mlirModuleOpaquePtr);
};

} // namespace cudaq
