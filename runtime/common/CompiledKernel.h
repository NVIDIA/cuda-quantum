/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/ThunkInterface.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// This header file and the types defined within are designed to have no
// dependencies and be useable across the compiler and runtime. However,
// constructing instances of these types is easiest done within compilation
// units that do link against MLIR. We provide this functionality via free
// functions, defined as friends of the types defined here and implemented in
// the `cudaq-mlir-runtime` library.

namespace mlir {
class Type;
class ModuleOp;
class ExecutionEngine;
} // namespace mlir

namespace cudaq {
class ResultInfo;
} // namespace cudaq

namespace cudaq_internal::compiler {
cudaq::ResultInfo createResultInfo(mlir::Type resultType, bool isEntryPoint,
                                   mlir::ModuleOp module);
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
/// for result buffer allocation and type conversion. Construct via
/// `createResultInfo` (implemented in `cudaq-mlir-runtime`).
class ResultInfo {
  // Friend factory function, to be used for construction.
  friend cudaq::ResultInfo cudaq_internal::compiler::createResultInfo(
      mlir::Type resultType, bool isEntryPoint, mlir::ModuleOp module);
  friend class CompiledKernel;

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

/// @brief A compiled, ready-to-execute kernel.
///
/// Bundles one or more representations of a compiled kernel (JIT binary, MLIR
/// module) along with metadata needed for execution and result extraction.
///
/// This type does not have a dependency on MLIR (or LLVM) as it only keeps
/// type-erased / opaque pointers. Use `attachJit` (defined in
/// `cudaq_internal/compiler/JIT.h`) to attach a compiled JIT representation
/// after construction.
class CompiledKernel {
public:
  // --- Construction ---

  CompiledKernel(std::string kernelName, ResultInfo resultInfo);

  // --- Queries ---

  bool hasJit() const { return jitRepr.has_value(); }
  bool hasMlir() const { return mlirRepr.has_value(); }

  /// Whether the kernel is fully specialized (all arguments inlined). For JIT
  /// kernels this means `argsCreator` is null.
  /// Currently, MLIR-only kernels are always considered fully specialized.
  bool isFullySpecialized() const {
    return !jitRepr || jitRepr->argsCreator == nullptr;
  }

  const std::string &getName() const { return name; }

  // --- Execution (local JIT path) ---

  /// @brief Execute a fully specialized kernel (no external arguments needed).
  KernelThunkResultType execute() const;

  /// @brief Execute the JIT-ed kernel with caller-provided arguments.
  KernelThunkResultType execute(const std::vector<void *> &rawArgs) const;

  // TODO: remove the following two methods once the `CompiledKernel` is
  // returned to Python.

  /// @brief Get the entry point of the kernel as a function pointer.
  ///
  /// The returned function pointer will expect different arguments depending
  /// on the kernel:
  ///  - if the kernel returns a value and/or is not fully specialized, the
  ///    entry point will expect a pointer to a buffer storing the packed
  ///    arguments and result.
  ///  - otherwise, the entry point will not expect any arguments.
  ///
  /// Prefer using `CompiledKernel::execute` instead of calling this function as
  /// it will handle the buffer and argument packing automatically.
  void (*getEntryPoint() const)();
  JitEngine getEngine() const;

  /// @brief Populate the JIT representation of a `CompiledKernel`.
  ///
  /// Resolves the entry point and (optionally) `argsCreator` symbols from the
  /// engine, using the kernel's name and result metadata to determine the
  /// correct mangled symbol names.
  void attachJit(JitEngine engine, bool isFullySpecialized);

private:
  // --- Compiled representation formats ---

  /// JIT-compiled representation of a kernel, used for local execution.
  struct JitRepr {
    JitEngine engine;
    void (*entryPoint)() = nullptr;
    int64_t (*argsCreator)(const void *, void **) = nullptr;
  };

  /// MLIR module representation for remote code generation or re-targeting.
  /// The opaque pointer is obtained via `ModuleOp::getAsOpaquePointer()`.
  /// Lifetime: the `MLIRContext` that owns the module must outlive this object.
  struct MlirRepr {
    const void *modulePtr = nullptr;
  };

  const JitRepr &getJit() const;
  const MlirRepr &getMlir() const;

  std::string name;
  ResultInfo resultInfo; // TODO: we might want to store the entire kernel
                         // signature here. Though I'm not sure what MLIR
                         // agnostic information is worth storing.
  std::optional<JitRepr> jitRepr;
  std::optional<MlirRepr> mlirRepr;
};

} // namespace cudaq
