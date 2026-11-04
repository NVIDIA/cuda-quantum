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
#include <vector>

namespace cudaq {
struct KernelExecution;
} // namespace cudaq

namespace mlir {
class Type;
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace cudaq_internal::compiler {

class Compiler;

/// Compiler-side helper for `cudaq::CompiledModule`: static factory methods and
/// utilities that depend on MLIR but pair with the MLIR-free `CompiledModule`
/// API in `common/CompiledModule.h`.
class CompiledModuleHelper {
public:
  // --- Named artifact aliases ---

  using NamedCompiledArtifact =
      std::pair<std::string, cudaq::CompiledModule::CompiledArtifact>;

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
  /// mangled symbol names.
  static std::vector<NamedCompiledArtifact>
  createJitArtifacts(const std::string &kernelName, cudaq::JitEngine engine,
                     const cudaq::ResultInfo &resultInfo,
                     bool isFullySpecialized);

  // --- ResourcesArtifact construction ---

  /// Construct a named `ResourcesArtifact` from pre-computed resource counts.
  static NamedCompiledArtifact createResourcesArtifact(std::string name,
                                                       cudaq::Resources rc);

  // --- CodegenArtifact construction ---

  /// Construct a named `CodegenArtifact` from emitted target code and its
  /// JSON-serialized output names.
  static NamedCompiledArtifact createCodegenArtifact(std::string name,
                                                     std::string code,
                                                     std::string outputNames);

  // --- MlirArtifact construction and access ---

  /// Construct a named `MlirArtifact` from a `ModuleOp`.
  static NamedCompiledArtifact
  createMlirArtifact(std::string name, mlir::ModuleOp module,
                     std::shared_ptr<mlir::MLIRContext> context = nullptr);

  /// Extract the `ModuleOp` from a `MlirArtifact`.
  static mlir::ModuleOp
  getMlirModuleOp(const cudaq::CompiledModule::MlirArtifact &artifact);

  // --- CompiledModule construction ---

  /// Create a `CompiledModule` containing the given compiled artifacts.
  static cudaq::CompiledModule createCompiledModule(
      std::string name, cudaq::ResultInfo resultInfo,
      std::vector<NamedCompiledArtifact> compiledArtifacts,
      cudaq::CompiledModule::CompilationMetadata metadata = {});

  /// Load the module's source MLIR using `Compiler::loadQuakeCodeByName` and
  /// return it.
  static cudaq::FatQuakeModule::MlirArtifact
  loadMlirArtifact(const cudaq::SourceModule &module);

  // --- KernelExecution construction ---

  /// Produce one `KernelExecution` per `MlirArtifact` of \p compiled, from the
  /// target code, JIT and resource artifacts emitted at compile time.
  ///
  /// If \p requireTargetCode is true, each MLIR artifact must have a
  /// `CodegenArtifact` with the same name, or an exception is thrown.
  static std::vector<cudaq::KernelExecution>
  toKernelExecutions(const cudaq::CompiledModule &compiled,
                     bool requireTargetCode = true);
};

} // namespace cudaq_internal::compiler
