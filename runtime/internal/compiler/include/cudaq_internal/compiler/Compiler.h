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
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cudaq {
class ServerHelper;
struct KernelExecution;
class ExecutionContext;
class noise_model;
namespace config {
class TargetConfig;
}
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

  /// @brief The Pass pipeline string, configured by the
  /// QPU configuration file in the platform path.
  std::string passPipelineConfig = "canonicalize";

  /// @brief Name of code generation target (e.g. `qir-adaptive`, `qir-base`,
  /// `qasm2`, `iqm`)
  std::string codegenTranslation = "";

  /// @brief Additional passes to run after the codegen-specific passes
  std::string postCodeGenPasses = "";

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
  std::pair<mlir::ModuleOp, mlir::func::FuncOp>
  prepareModule(const std::string &kernelName, mlir::ModuleOp m_module,
                cudaq::KernelArgs args);

  /// Delay combine-measurements for emulation, then run the main pass
  /// pipeline.  Returns true when combine-measurements was delayed.
  bool executeMainPipeline(mlir::ModuleOp moduleOp,
                           const std::string &kernelName);

  /// Create JIT and MLIR artifacts and assemble a CompiledModule.
  cudaq::CompiledModule assembleCompiledModule(
      const std::string &kernelName,
      std::vector<std::pair<std::string, mlir::ModuleOp>> &modules,
      bool needJit, bool runCombineMeasurements,
      std::optional<cudaq::Resources> resourceCounts,
      const std::vector<std::size_t> &mappingReorderIdx,
      std::shared_ptr<mlir::MLIRContext> context);

public:
  static std::pair<const void *, std::shared_ptr<mlir::MLIRContext>>
  loadQuakeCodeByName(const std::string &kernelName);

  Compiler(cudaq::ServerHelper *,
           const std::map<std::string, std::string> &backendConfig,
           cudaq::config::TargetConfig &config,
           const cudaq::noise_model *noiseModel, bool emulate);
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
  runPassPipeline(cudaq::ExecutionContext *executionContext,
                  const std::string &kernelName, const void *modulePtr,
                  cudaq::KernelArgs args,
                  std::shared_ptr<mlir::MLIRContext> context = nullptr);

  /// @brief Emit target-specific code for each `MlirArtifact` in the
  /// `CompiledModule` and produce `KernelExecution` objects.
  std::vector<cudaq::KernelExecution>
  emitKernelExecutions(const cudaq::CompiledModule &compiled);

  /// Compile the quake code passed via ModuleOp and lower it to the code format
  /// required for the specific backend.
  ///
  /// The lowering process is controllable via the configuration file in the
  /// platform directory for the targeted backend.
  ///
  /// Unchecked assumption: there are no other references to \p module (within
  /// the scope of this launch instance). It can be disposed and/or modified by
  /// this call in any way necessary without breaking some other kernel launch.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(cudaq::ExecutionContext *executionContext,
                 const std::string &kernelName, const void *modulePtr,
                 cudaq::KernelArgs args);
};
} // namespace cudaq_internal::compiler
