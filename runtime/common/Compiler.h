/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "JIT.h"
#include "ServerHelper.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cudaq {
class ServerHElper;
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

namespace cudaq {

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

  /// @brief If we are emulating locally, keep track
  /// of JIT engines for invoking the kernels.
  std::vector<JitEngine> jitEngines;

  /// @brief Flag indicating whether we should emulate execution locally.
  bool emulate = false;

  /// @brief Flag indicating whether we should print the IR.
  bool printIR = false;

  std::vector<cudaq::KernelExecution>
  lowerQuakeCodePart2(const std::string &kernelName, void *kernelArgs,
                      const std::vector<void *> &rawArgs,
                      mlir::ModuleOp m_module, mlir::MLIRContext *contextPtr,
                      void *updatedArgs);

  std::tuple<mlir::ModuleOp, std::unique_ptr<mlir::MLIRContext>, void *>
  extractQuakeCodeAndContext(const std::string &kernelName, void *data);

public:
  Compiler(cudaq::ServerHelper *,
           const std::map<std::string, std::string> &backendConfig,
           config::TargetConfig &config, const noise_model *noiseModel,
           bool emulate);
  ~Compiler();

  mlir::ModuleOp lowerQuakeCodeBuildModule(const std::string &,
                                           mlir::ModuleOp module,
                                           mlir::MLIRContext *,
                                           mlir::func::FuncOp);

  /// @brief Extract the Quake representation for the given kernel name and
  /// lower it to the code format required for the specific backend. The
  /// lowering process is controllable via the configuration file in the
  /// platform directory for the targeted backend.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs,
                 const std::vector<void *> &rawArgs);

  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, void *kernelArgs);

  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName,
                 const std::vector<void *> &rawArgs);

  // Here the quake code is passed to us (via a ModuleOp), so unlike the other
  // lowerQuakeCode() member functions there is no need to surf dictionaries for
  // strings of code to assemble. We have to make sure that this MLIRContext is
  // not destroyed however, since it may hold an unknown number of other
  // ModuleOps.
  // Unchecked assumption: \p module is referentially unique (within the scope
  // of this launch instance) and disposable. It can be modified by this call in
  // any way necessary without breaking some other kernel launch.
  std::vector<cudaq::KernelExecution>
  lowerQuakeCode(const std::string &kernelName, mlir::ModuleOp module,
                 const std::vector<void *> &rawArgs);
};
} // namespace cudaq
