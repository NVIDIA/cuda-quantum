/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/CompiledModule.h"
#include <memory>

namespace mlir {
class Type;
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cudaq_internal::compiler {

/// Compiler-side helper for `cudaq::CompiledModule`: static factory methods and
/// utilities that depend on MLIR but pair with the MLIR-free `CompiledModule`
/// API in `common/CompiledModule.h`.
class CompiledModuleHelper {
public:
  // --- Named artifact aliases ---

  using NamedJitArtifact =
      std::pair<std::string, cudaq::CompiledModule::JitArtifact>;
  using NamedMlirArtifact =
      std::pair<std::string, cudaq::CompiledModule::MlirArtifact>;

  CompiledModuleHelper() = delete;

  // --- ResultInfo construction ---

  /// Create a `ResultInfo` from MLIR type metadata.
  ///
  /// When \p resultType is null or \p isEntryPoint is false, returns an empty
  /// `ResultInfo` (no marshaled return value).
  static cudaq::ResultInfo createResultInfo(mlir::Type resultType,
                                            bool isEntryPoint,
                                            mlir::ModuleOp module);

  // --- JitArtifact construction ---

  /// Construct named JitArtifacts from the compiled functions in the JIT
  /// engine.
  ///
  /// Uses the kernel's name and result metadata to determine the correct
  /// mangled symbol names. Currently returns one artifact.
  ///
  /// Optionally, a \p resourceCounts can be attached to the returned artifact.
  static std::vector<NamedJitArtifact> createJitArtifacts(
      const std::string &kernelName, cudaq::JitEngine engine,
      const cudaq::ResultInfo &resultInfo, bool isFullySpecialized,
      std::optional<cudaq::Resources> resourceCounts = std::nullopt);

  // --- MlirArtifact construction and access ---

  /// Construct a named `MlirArtifact` from a `ModuleOp`.
  static NamedMlirArtifact
  createMlirArtifact(std::string name, mlir::ModuleOp module,
                     std::shared_ptr<mlir::MLIRContext> context = nullptr);

  /// Extract the `ModuleOp` from a `MlirArtifact`.
  static mlir::ModuleOp
  getMlirModuleOp(const cudaq::CompiledModule::MlirArtifact &artifact);

  // --- CompiledModule construction ---

  /// Create a `CompiledModule` containing only JIT artifacts.
  static cudaq::CompiledModule createCompiledModule(
      std::string name, cudaq::ResultInfo resultInfo,
      std::vector<NamedJitArtifact> jitArtifacts,
      cudaq::CompiledModule::CompilationMetadata metadata = {});

  /// Create a `CompiledModule` containing only MLIR artifacts.
  static cudaq::CompiledModule createCompiledModule(
      std::string name, cudaq::ResultInfo resultInfo,
      std::vector<NamedMlirArtifact> mlirArtifacts,
      cudaq::CompiledModule::CompilationMetadata metadata = {});

  /// Create a `CompiledModule` containing both JIT and MLIR artifacts.
  static cudaq::CompiledModule createCompiledModule(
      std::string name, cudaq::ResultInfo resultInfo,
      std::vector<NamedJitArtifact> jitArtifacts,
      std::vector<NamedMlirArtifact> mlirArtifacts,
      cudaq::CompiledModule::CompilationMetadata metadata = {});
};

} // namespace cudaq_internal::compiler
