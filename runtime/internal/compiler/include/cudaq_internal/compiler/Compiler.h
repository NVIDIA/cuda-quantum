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
#include "cudaq/Optimizer/CodeGen/CodegenMode.h"
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

/// A target's JIT pass pipeline.
struct PassPipeline {
  std::string text;
  /// Whether every stage has `StagePasses::normalizesKernel` set, so that
  /// `codegen` only emits. A pipeline that ran no stage does not count.
  bool everyStageNormalizesKernel = false;
};

/// How much of the kernel's preparation `codegen` still does for a kernel
/// that was compiled with \p metadata. When every stage normalized the
/// kernel, `codegen` only emits. Otherwise `codegen` normalizes the kernel
/// itself and, above level 0, also optimizes it. That includes a compile that
/// skipped the target pipeline, since then no stage ran.
// TODO: REST targets take the level from `codegen` until their target
// definitions declare a stage that applies it.
cudaq::opt::CodegenMode
codegenMode(const cudaq::CompiledModule::CompilationMetadata &metadata);

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

  // ---- Common helpers used by `compile` ----

  /// Run an arbitrary MLIR pass pipeline string on a module.
  void applyPipeline(const std::string &pipeline, mlir::ModuleOp moduleOp,
                     const std::string &kernelName);

  /// Build the module, merge closures, synthesize arguments, and erase unused
  /// private symbols. Returns the module, the entry point, and whether every
  /// argument was specialized.
  std::tuple<mlir::ModuleOp, mlir::func::FuncOp, bool>
  prepareModule(const std::string &kernelName, mlir::ModuleOp m_module,
                cudaq::KernelArgs args, bool isEntryPoint);

  /// Generate the host functions that marshal the kernel's arguments and
  /// result.
  void generateHostFunctions(mlir::ModuleOp moduleOp, bool isFullySpecialized);

  /// Delay combine-measurements for emulation, then run the main pass
  /// pipeline built from \p resolvedOptions, the options resolved for this
  /// compile.  Returns
  ///  - whether combine-measurements was delayed,
  ///  - the pass pipeline that was executed, which is empty when none ran.
  std::pair<bool, PassPipeline>
  executeMainPipeline(mlir::ModuleOp moduleOp, const std::string &kernelName,
                      const cudaq::CompileOptions &resolvedOptions);

  /// What `assembleCompiledModule` produces besides the MLIR artifact.
  struct AssemblyRequest {
    /// JIT-compile the kernel.
    bool needsJit = false;
    /// Give the module that the JIT compiles the host functions that marshal
    /// the kernel's arguments and result.
    bool needsHostFunctions = false;
    /// Every argument of the kernel was specialized.
    bool isFullySpecialized = false;
    /// The kernel is a complete execution boundary, not a nested callable.
    bool isEntryPoint = false;
    /// Run `combine-measurements` on the modules after the JIT has its copy.
    bool combineMeasurements = false;
  };

  /// Create JIT and MLIR artifacts and assemble a CompiledModule.
  cudaq::CompiledModule assembleCompiledModule(
      const std::string &kernelName,
      std::vector<std::pair<std::string, mlir::ModuleOp>> &modules,
      const AssemblyRequest &request,
      std::optional<cudaq::Resources> resourceCounts,
      cudaq::CompiledModule::CompilationMetadata metadata,
      std::shared_ptr<mlir::MLIRContext> context);

  /// Translate \p module according to the target's `codegen` emission format.
  /// Returns the target code and its JSON-serialized output names mapping.
  std::pair<std::string, std::string>
  emitTargetCode(mlir::ModuleOp module, cudaq::opt::CodegenMode codegenMode);

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
  /// Performs argument synthesis, the full pass pipeline, observation
  /// splitting (for observe mode) and `codegen`.
  ///
  /// If \p context is provided, `module.getContext() == context.get()` must
  /// be true. In that case, the MLIR artifacts will keep a `shared_ptr` to
  /// the context, guaranteeing it outlives the artifacts. Otherwise the
  /// context lifetime must be managed by the caller.
  cudaq::CompiledModule
  compile(const std::string &kernelName, const void *modulePtr,
          cudaq::KernelArgs args, bool isEntryPoint,
          std::shared_ptr<mlir::MLIRContext> context = nullptr);
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
/// deploy, and finalize. When no stage needs gate lowering, prep leaves the
/// gates that hardware lacks in the kernel and deploy does not run. When every
/// stage also normalizes the kernel, prep and finalize skip the work that the
/// stages' `prepare-for-wireset` repeats. If the target
/// sets `disable_qubit_mapping`, every `qubit-mapping` in the result gets
/// `device=bypass`.
std::optional<PassPipeline>
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
