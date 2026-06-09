/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/CompiledModule.h"
#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/algorithms/sample/policy.h"
#include <memory>
#include <string>
#include <vector>

namespace cudaq {
struct KernelExecution;
class ExecutionContext;
} // namespace cudaq

namespace mlir {
class ModuleOp;
class MLIRContext;
class ExecutionEngine;

namespace func {
class FuncOp;
}
} // namespace mlir

namespace cudaq_internal::compiler {

class Compiler {
  /// @brief Flag indicating whether we should perform the passes in a
  /// single-threaded environment, useful for debug. Similar to
  /// `-mlir-disable-threading` for `cudaq-opt`.
  bool disableMLIRthreading = false;

  /// @brief Flag indicating whether we should enable MLIR printing before and
  /// after each pass. This is similar to `-mlir-print-ir-before-all` and
  /// `-mlir-print-ir-after-all` in `cudaq-opt`.
  bool enablePrintMLIREachPass = false;

  /// @brief Flag indicating whether we should enable MLIR pass statistics
  /// to be printed. This is similar to `-mlir-pass-statistics` in `cudaq-opt`
  bool enablePassStatistics = false;

  /// @brief Flag indicating whether we should emulate execution locally.
  bool emulate = false;

  /// @brief The compile target configuration containing the compile options.
  std::unique_ptr<cudaq::CompileTarget> target;

  /// @brief Flag indicating whether we should print the IR.
  bool printIR = false;

  /// Whether compilation emitted a named measurement warning.
  bool warnedNamedMeasurements = false;

  mlir::ModuleOp lowerQuakeCodeBuildModule(const std::string &,
                                           mlir::ModuleOp module,
                                           mlir::MLIRContext *,
                                           mlir::func::FuncOp);

  // ---- Common helpers used by runPassPipeline ----

  /// Run an arbitrary MLIR pass pipeline string on a module.
  void applyPipeline(const std::string &pipeline, mlir::ModuleOp moduleOp,
                     const std::string &kernelName);

  /// Build the module, merge closures, and synthesize arguments.
  std::tuple<mlir::ModuleOp, mlir::func::FuncOp, bool>
  prepareModule(const std::string &kernelName, mlir::ModuleOp m_module,
                cudaq::KernelArgs args, bool isEntryPoint);

  /// Delay combine-measurements for emulation, then run the main pass
  /// pipeline.  Returns
  ///  - whether combine-measurements was delayed,
  ///  - the pass pipeline that was executed.
  std::pair<bool, std::string>
  executeMainPipeline(mlir::ModuleOp moduleOp, const std::string &kernelName);

  /// Create JIT and MLIR artifacts and assemble a CompiledModule.
  cudaq::CompiledModule assembleCompiledModule(
      const std::string &kernelName,
      std::vector<std::pair<std::string, mlir::ModuleOp>> &modules,
      bool needJit, bool isFullySpecialized, bool isEntryPoint,
      bool runCombineMeasurements,
      std::optional<cudaq::Resources> resourceCounts,
      cudaq::CompiledModule::CompilationMetadata metadata,
      std::shared_ptr<mlir::MLIRContext> context);

public:
  /// Whether compilation emitted a warning about the presence of named
  /// measurements.
  bool hasWarnedNamedMeasurements() const { return warnedNamedMeasurements; }

  const cudaq::CompileTarget &getTarget() const { return *target; }

  static std::pair<const void *, std::shared_ptr<mlir::MLIRContext>>
  loadQuakeCodeByName(const std::string &kernelName);

  Compiler(std::unique_ptr<cudaq::CompileTarget> &&target);
  ~Compiler();

  /// @brief Compile the given module and return a `CompiledModule`.
  ///
  /// Performs argument synthesis, the full pass pipeline, and observation
  /// splitting (for observe mode).
  ///
  /// If \p context is provided, `module.getContext() == context.get()` must
  /// be true. In that case, the MLIR artifacts will keep a `shared_ptr` to
  /// the context, guaranteeing it outlives the artifacts. Otherwise the
  /// context lifetime must be managed by the caller.
  cudaq::CompiledModule
  runPassPipeline(const std::string &kernelName, const void *modulePtr,
                  cudaq::KernelArgs args, bool isEntryPoint,
                  std::shared_ptr<mlir::MLIRContext> context = nullptr);

  /// @brief Emit target-specific code for each `MlirArtifact` in the
  /// `CompiledModule` and produce `KernelExecution` objects.
  std::vector<cudaq::KernelExecution>
  emitKernelExecutions(const cudaq::CompiledModule &compiled);
};

/// Get the pass pipeline string for the given compile target.
///
/// If `target.pipelineConfig.overridePassPipeline` is set, returns it directly
/// (full override, interleave stages are ignored). Otherwise builds: [high]
/// [,deployStage] [,mid] [,finalizeStage] [,low] where deployStage and
/// finalizeStage are fixed stages interleaved between the config-provided
/// stages. Pass empty strings to skip them.
std::string getPassPipeline(const cudaq::CompileTarget &target);

/// Compile a source module for the given policy, compile target and
/// arguments.
template <typename Policy>
cudaq::CompiledModule
compileModule(Policy *policy, std::unique_ptr<cudaq::CompileTarget> target,
              const cudaq::SourceModule &src, cudaq::KernelArgs args,
              bool isEntryPoint = true) {
  const auto &kernelName = src.getName();
  auto modulePtr = src.getMlirOpaqueModulePtr();
  assert(modulePtr && "Compiler::compileModule requires an MLIR artifact");

  Compiler compiler(std::move(target));
  auto compiled =
      compiler.runPassPipeline(kernelName, modulePtr, args, isEntryPoint);

  if constexpr (std::is_same_v<Policy, cudaq::sample_policy>) {
    if (compiler.hasWarnedNamedMeasurements())
      policy->warnedNamedMeasurements = true;
  }
  return compiled;
}

} // namespace cudaq_internal::compiler
