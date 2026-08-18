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
struct CompileTarget {
  /// Whether to recompile the kernel in the presence of an AOT-compiled module.
  ///
  /// If this is `false` and an AOT-compiled kernel (in the form of a function
  /// pointer) is provided, then compilation will be skipped and all other
  /// options in this class will be ignored.
  ///
  /// If this is `true`, the AOT-compiled module (if it exists) will be
  /// discarded and compilation will start from scratch, according to the
  /// options in this class.
  bool overrideAOTCompilation = false;

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

    /// Whether to disable qubit mapping.
    bool disableQubitMapping = false;

    /// Whether to run the replace-state-with-kernel pass.
    ///
    /// Allows targets that do not support get_state (e.g. remote QPUs)
    /// to emulate its behavior by inserting the corresponding kernel calls.
    bool replaceStateWithKernel = false;

    /// Whether the pipeline is empty.
    bool empty() const {
      return overridePassPipeline.empty() && highLevelPipeline.empty() &&
             midLevelPipeline.empty() && lowLevelPipeline.empty();
    }

    bool operator==(const PipelineConfig &other) const = default;
  };

  /// Pipeline configuration, populated by the constructor.
  PipelineConfig pipelineConfig;

  /// Whether branching on measurement results is supported.
  bool supportConditionalsOnMeasureResults = true;

  /// Whether device calls are supported by the target.
  bool supportDeviceCalls = false;

  /// Whether explicit measurements are supported by the target.
  bool supportExplicitMeasurements = true;

  /// Whether the target supports measuring arbitrary observables.
  ///
  /// If false, the compiler will implement observable measurements by splitting
  /// the hamiltonian into a sum of Pauli measurements and emit one lowered
  /// module per Pauli term.
  bool supportObservableMeasurements = true;

  /// Whether the target supports sampling without explicit measurements in the
  /// IR.
  ///
  /// If false, when compiling for `sample`, the compiler will ensure explicit
  /// measurements are present at the end of the kernel (or add them if not).
  bool supportSampleWithoutMeasurements = true;

  /// Whether to fully specialize the kernel.
  bool fullySpecialize = true;

  /// Set the `changeSemantics` flag for the argument synthesis pass.
  bool argumentSynthChangeSemantics = true;

  /// Construct a CompileTarget from static and runtime backend configurations.
  CompileTarget(config::TargetConfig targetConfig,
                std::map<std::string, std::string> runtimeConfig,
                std::map<std::string, std::string> pipelineSubstitutions = {});

  CompileTarget() = default;

  bool operator==(const CompileTarget &other) const = default;
};

} // namespace cudaq

template <>
struct std::hash<cudaq::CompileTarget> {
  std::size_t operator()(const cudaq::CompileTarget &t) const noexcept;
};
template <>
struct std::hash<cudaq::CompileTarget::PipelineConfig> {
  std::size_t
  operator()(const cudaq::CompileTarget::PipelineConfig &pc) const noexcept;
};
