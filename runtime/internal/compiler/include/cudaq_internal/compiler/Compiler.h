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

/// Build the JIT pass pipeline for \p target. On failure, set \p diagnostic to
/// say which stage failed and why, and return `nullopt`.
///
/// Each stage can add passes before deploy, after deploy, and after finalize
/// (see `StagePasses`). Passes that different stages add at the same place run
/// in stage order. For example, if the first stage adds `A` before deploy and
/// `B` after deploy, and the second stage adds `C` after deploy and `D` after
/// finalize, the pipeline is
/// `canonicalize, prep, A, deploy, B, C, finalize, D`.
///
/// A stage's lowering decides its passes:
///  - An inferred lowering uses the first handler in \p handlers that can
///    lower the previous stage's output profile (`unchecked` for the first
///    stage) to this stage's output profile.
///  - A custom lowering puts its high-, mid-, and low-level pipelines before
///    deploy, after deploy, and after finalize.
///  - An override pipeline replaces the whole pipeline, so it must be the
///    target's only stage.
///
/// The emission format and `CompileOptions::emulate` set the options of prep,
/// deploy, and finalize. If the target sets `disable_qubit_mapping`, every
/// `qubit-mapping` in the result gets `device=bypass`.
std::optional<std::string>
getPassPipeline(const cudaq::CompileTarget &target,
                const cudaq::CompileOptions &options,
                const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
                std::string &diagnostic);

/// Whether compiles for \p target substitute the values of an entry point's
/// arguments instead of taking them at run time. That holds when the platform
/// cannot take arguments at run time (`fullySpecialize`) or when the handler
/// that `getPassPipeline` selects for an inferred stage requires argument
/// values. Custom and override stages are opaque, so a target whose custom
/// pipeline needs values sets `fullySpecialize`.
///
/// The compiler, the Python compiled-module cache, and Python argument packing
/// all read this answer, and they must agree, because the arguments are laid
/// out differently for the two cases. It depends only on the target so that
/// Python can pack arguments before a launch's `CompileOptions` exist.
bool specializesArguments(
    const cudaq::CompileTarget &target,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers);

/// `specializesArguments` with the built-in lowering handlers.
bool specializesArguments(const cudaq::CompileTarget &target);

/// Compile a source module for the given policy, compile target and
/// arguments.
cudaq::CompiledModule compileModule(cudaq::CompileTarget target,
                                    cudaq::CompileOptions options,
                                    const cudaq::SourceModule &src,
                                    cudaq::KernelArgs args,
                                    bool isEntryPoint = true);

} // namespace cudaq_internal::compiler
