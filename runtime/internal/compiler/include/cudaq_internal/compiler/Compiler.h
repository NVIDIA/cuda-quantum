/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/CompileOptions.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/algorithms/sample/policy.h"
#include "cudaq/runtime/logger/logger.h"
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

  /// @brief Whether to enable MLIR printing before and after each pass.
  ///
  /// This is similar to `-mlir-print-ir-before-all` and
  /// `-mlir-print-ir-after-all` in `cudaq-opt`. Printing can be enabled for all
  /// passes or just during specialization.
  cudaq::PrintEachPassMode printEachPass = cudaq::PrintEachPassMode::None;

  /// @brief Flag indicating whether we should enable MLIR pass statistics
  /// to be printed. This is similar to `-mlir-pass-statistics` in `cudaq-opt`
  bool enablePassStatistics = false;

  /// @brief Flag indicating whether we should emulate execution locally.
  bool emulate = false;

  /// @brief The compile target describing the architecture to compile for.
  cudaq::CompileTarget target;

  /// @brief The compile options to compile with.
  cudaq::CompileOptions options;

  /// @brief The lowering handlers, consulted in order to infer compilation
  /// pipelines.
  std::vector<std::unique_ptr<LoweringHandler>> handlers;

  /// @brief Flag indicating whether we should print the IR.
  bool printIR = false;

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
  /// pipeline built from \p resolvedOptions, the options resolved for this
  /// compile.  Returns
  ///  - whether combine-measurements was delayed,
  ///  - the pass pipeline that was executed.
  std::pair<bool, std::string>
  executeMainPipeline(mlir::ModuleOp moduleOp, const std::string &kernelName,
                      const cudaq::CompileOptions &resolvedOptions);

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
  const cudaq::CompileTarget &getTarget() const { return target; }
  const cudaq::CompileOptions &getOptions() const { return options; }

  static std::pair<const void *, std::shared_ptr<mlir::MLIRContext>>
  loadQuakeCodeByName(const std::string &kernelName);

  /// @brief Construct with the default set of lowering handlers.
  Compiler(const cudaq::CompileTarget &target,
           const cudaq::CompileOptions &options);

  /// @brief Construct with an explicit, complete set of lowering handlers.
  Compiler(const cudaq::CompileTarget &target,
           const cudaq::CompileOptions &options,
           std::vector<std::unique_ptr<LoweringHandler>> handlers);
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

/// Get the pass pipeline for the given compile target, resolving default
/// (inferred) stages through \p handlers.
///
/// An override pipeline is the whole pipeline, so it must be the target's only
/// stage. Otherwise the pipeline runs `canonicalize`, prep, deploy, and
/// finalize once, with options set from the emission format and
/// `CompileOptions::emulate`, and joins the stages' passes around them in
/// stage order. A `CustomPipelineLowering` supplies its high-, mid-, and
/// low-level pipelines. An `InferredLowering` takes the `StagePasses` of the
/// first handler in \p handlers that supports lowering from the previous
/// stage's output profile, or `unchecked` for the first stage, to this stage's
/// output profile. When the target sets `disable_qubit_mapping`, every
/// `qubit-mapping` in the result, including one in an override, gets
/// `device=bypass`. Fills \p diagnostic when no handler lowers a stage, when an
/// override is not the only stage, or when a stage sets passes that
/// `StagePasses` reserves for the first or last stage.
std::optional<std::string>
getPassPipeline(const cudaq::CompileTarget &target,
                const cudaq::CompileOptions &options,
                const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
                std::string &diagnostic);

/// Compile a source module for the given policy, compile target and
/// arguments.
cudaq::CompiledModule compileModule(cudaq::CompileTarget target,
                                    cudaq::CompileOptions options,
                                    const cudaq::SourceModule &src,
                                    cudaq::KernelArgs args,
                                    bool isEntryPoint = true);

} // namespace cudaq_internal::compiler
