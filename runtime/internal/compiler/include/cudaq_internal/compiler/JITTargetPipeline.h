/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/CompileTarget.h"
#include "mlir/Pass/PassManager.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
struct CompileOptions;
}

namespace cudaq_internal::compiler {

/// The JIT pass pipeline of a compile target, as built by `getPassPipeline`.
struct JITPassPipeline {
  /// The passes to run, anchored on `builtin.module`.
  mlir::OpPassManager passes{"builtin.module"};

  /// Whether `combine-measurements` should be run after building the JIT code.
  ///
  /// Set when the target's custom pipelines request the pass and the compiler
  /// chose to defer it (either to emulate or until observe splits the kernel
  /// into terms)
  bool deferCombineMeasurements = false;

  /// The gate set mapping passes.
  ///
  /// Saved here as observe reruns them on each term's module.
  mlir::OpPassManager gateSetMappingPasses{"builtin.module"};
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
/// deploy, and finalize. If the target sets `disable_qubit_mapping`, mapping
/// routes onto the `bypass` device: an inferred `quake-mapped` stage maps to
/// it, and every `qubit-mapping` in custom text gets `device=bypass`.
///
/// Custom pipeline text from the target configuration is parsed here.
std::optional<JITPassPipeline>
getPassPipeline(const cudaq::CompileTarget &target,
                const cudaq::CompileOptions &options,
                const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
                std::string &diagnostic);

/// The collected pipelines and options, ready to be converted into the final
/// JIT pass pipeline.
///
/// Useful for inspection and testing before converting into the final JIT pass
/// pipeline.
struct JITPipelineLayout {
  cudaq::opt::JITPrepPipelineOptions prep;
  cudaq::opt::JITDeployPipelineOptions deploy;
  cudaq::opt::JITFinalizePipelineOptions finalize;

  /// What each stage adds before deploy, after deploy, and after finalize, in
  /// stage order (see `StagePasses`).
  std::vector<StagePasses> stages;

  /// As in `JITPassPipeline`.
  bool deferCombineMeasurements = false;
  mlir::OpPassManager gateSetMappingPasses{"builtin.module"};

  /// Append the JIT pass pipeline to \p pm: `canonicalize`, prep, the stages'
  /// passes before deploy, deploy, their passes after deploy, finalize, and
  /// their passes after finalize.
  void addPasses(mlir::OpPassManager &pm) const;

  /// Append the passes that every stage adds at \p place to \p pm, in stage
  /// order.
  void addStagePasses(mlir::OpPassManager &pm,
                      AddPassesFn StagePasses::*place) const;
};

/// Collect the inferred pipelines from the compilation stages of \p target.
/// Custom pipeline text is checked here, so a stage whose text does not parse
/// fails with its stage number. A target without stages, or with an override
/// pipeline, has no layout; for those, set \p diagnostic and return
/// `nullopt`.
std::optional<JITPipelineLayout>
getPipelineLayout(const cudaq::CompileTarget &target,
                  const cudaq::CompileOptions &options,
                  const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
                  std::string &diagnostic);

} // namespace cudaq_internal::compiler
