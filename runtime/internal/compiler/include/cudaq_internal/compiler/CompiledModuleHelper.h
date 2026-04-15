/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution. *
 ******************************************************************************/
#pragma once

#include "common/CompiledModule.h"

namespace mlir {
class Type;
class ModuleOp;
} // namespace mlir

namespace cudaq_internal::compiler {

/// Compiler-library support for `cudaq::CompiledModule`: builders, layout, and
/// other types that must depend on MLIR but pair with the MLIR-free
/// `CompiledModule` API in `common/CompiledModule.h`.

/// Builder for constructing `CompiledModule` artifacts using MLIR. Once
/// `finish()` returns, the module is immutable and does not depend on MLIR.
///
/// Typical usage: construct with the kernel name, call `setResultInfo` when the
/// kernel's return value must be marshaled, then `attachJit`, then `finish()`.
class CompiledModuleBuilder {
  cudaq::CompiledModule compiled;

public:
  explicit CompiledModuleBuilder(std::string kernelName);

  /// @brief Pre-compute result buffer metadata from the kernel's MLIR return
  /// type.
  ///
  /// When \p resultType is null or \p isEntryPoint is false, clears result
  /// metadata (no marshaled return). Otherwise fills sizes and field offsets
  /// from the module data layout. Call before `attachJit` when symbol names
  /// depend on whether a result is present.
  void setResultInfo(mlir::Type resultType, bool isEntryPoint,
                     mlir::ModuleOp module);

  /// @brief Populate the JIT representation.
  ///
  /// Resolves the entry point and (optionally) `argsCreator` symbols from the
  /// engine, using the kernel's name and result metadata to determine the
  /// correct mangled symbol names.
  void attachJit(cudaq::JitEngine engine, bool isFullySpecialized);

  /// Release ownership of the built CompiledModule.
  cudaq::CompiledModule finish() { return std::move(compiled); }
};

} // namespace cudaq_internal::compiler
