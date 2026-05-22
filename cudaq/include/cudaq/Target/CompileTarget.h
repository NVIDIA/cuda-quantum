/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "cudaq/Target/TargetConfig.h"
#include "cudaq/operators.h"
#include <map>
#include <optional>
#include <string>

namespace cudaq {

/// Target properties used to define the compilation pipeline.
class CompileTarget {
public:
  /// Hook to update the pass pipeline before compilation.
  virtual void updatePassPipeline(std::string &passPipeline) const {}

  /// Resolved MLIR pass-pipeline and `codegen` settings.
  struct PipelineConfig {
    /// If set, override compilation pipeline with this string.
    std::string overridePassPipeline;

    /// Compilation pipeline to insert at the appropriate stages.
    std::string highLevelPipeline;
    std::string midLevelPipeline;
    std::string lowLevelPipeline;

    /// Code generation emission selected by the target.
    std::string codegenTranslation;

    /// Optional pass pipeline to run after code generation.
    std::string postCodeGenPasses;

    /// Backend data provides target pass pipeline fields.
    bool hasConfiguredPassPipeline = false;
    /// Standard `jit-finalize-pipeline` is part of `passPipeline`.
    bool runsStandardFinalize = false;
    /// Whether to disable qubit mapping.
    bool disableQubitMapping = false;
  };

  /// Pipeline configuration, populated by the constructor.
  PipelineConfig pipelineConfig;

  /// Whether to emulate execution locally.
  bool emulate = false;

  /// Issue a warning if named measurements are contained in the kernel.
  bool warnNamedMeasurements = false;

  /// Whether branching on measurement results is supported.
  bool supportConditionalsOnMeasureResults = true;

  /// Whether to retrieve mapping reorder indices from MLIR and store it as
  /// compiled metadata.
  bool storeReorderIdx = false;

  /// Whether to generate resource counts.
  ///
  /// When true, the compiler will generate resource counts during compilation
  /// and simplify the IR to remove all quantum operations already accounted
  /// for in the counts.
  bool generateResourceCounts = false;

  /// When set, emit one lowered module per non-identity Pauli term of this
  /// observable. The resulting `CompiledModule` will contain a compilation
  /// artifact for each term.
  std::optional<cudaq::spin_op> pauliTermSplitObservable;

  /// Construct a CompileTarget from static and runtime backend configurations.
  CompileTarget(config::TargetConfig targetConfig,
                std::map<std::string, std::string> runtimeConfig, bool emulate);

  CompileTarget() = default;
  CompileTarget(const CompileTarget &) = default;
  CompileTarget(CompileTarget &&) = default;
  CompileTarget &operator=(const CompileTarget &) = default;
  CompileTarget &operator=(CompileTarget &&) = default;
  virtual ~CompileTarget() = default;
};

} // namespace cudaq
