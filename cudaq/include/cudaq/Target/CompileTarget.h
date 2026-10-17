/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/CompilationStage.h"
#include <map>
#include <string>
#include <vector>

namespace cudaq::config {
class TargetConfig;
} // namespace cudaq::config

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

  /// Ordered compilation stages that the source program must be lowered
  /// through.
  std::vector<config::CompilationStage> stages{};

  /// Code generation configuration, running after all compilation stages.
  config::Codegen codegen{};

  /// Whether branching on measurement results is supported.
  bool supportConditionalsOnMeasureResults = true;

  /// Whether device calls are supported by the target.
  bool supportDeviceCalls = false;

  /// Whether the target supports pre-computing resource counts for faster
  /// resource estimation.
  ///
  /// Only used for resource estimation. When set, the compiler will strip out
  /// all gates outside of control flow and produce gate counts statically.
  bool supportResourceCounts = true;

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

  /// Whether to disable qubit mapping.
  bool disableQubitMapping = false;

  /// Whether to run the replace-state-with-kernel pass.
  ///
  /// Allows targets that do not support get_state (e.g. remote QPUs)
  /// to emulate its behavior by inserting the corresponding kernel calls.
  bool replaceStateWithKernel = false;

  /// If the entire compilation is given by a single custom pipeline, return a
  /// pointer to it. Otherwise return `nullptr`.
  const std::string *getOverridePassPipeline() const;

  /// Whether any of the compilation stages have a non-empty lowering pipeline.
  bool hasPassPipeline() const;

  /// Construct a CompileTarget from static and runtime backend configurations.
  static CompileTarget createFromConfig(
      config::TargetConfig targetConfig,
      std::map<std::string, std::string> runtimeConfig,
      std::map<std::string, std::string> pipelineSubstitutions = {});

  bool operator==(const CompileTarget &other) const = default;
};

} // namespace cudaq

template <>
struct std::hash<cudaq::config::UncheckedProfile> {
  std::size_t
  operator()(const cudaq::config::UncheckedProfile &profile) const noexcept;
};
template <>
struct std::hash<cudaq::config::QuakeReferenceSemanticsProfile> {
  std::size_t operator()(const cudaq::config::QuakeReferenceSemanticsProfile
                             &profile) const noexcept;
};
template <>
struct std::hash<cudaq::config::CustomPipelineLowering> {
  std::size_t operator()(
      const cudaq::config::CustomPipelineLowering &lowering) const noexcept;
};
template <>
struct std::hash<cudaq::config::DefaultLowering> {
  std::size_t
  operator()(const cudaq::config::DefaultLowering &lowering) const noexcept;
};
template <>
struct std::hash<cudaq::config::Codegen> {
  std::size_t operator()(const cudaq::config::Codegen &codegen) const noexcept;
};
template <>
struct std::hash<cudaq::config::CompilationStage> {
  std::size_t
  operator()(const cudaq::config::CompilationStage &stage) const noexcept;
};
template <>
struct std::hash<std::vector<cudaq::config::CompilationStage>> {
  std::size_t operator()(const std::vector<cudaq::config::CompilationStage>
                             &stages) const noexcept;
};
template <>
struct std::hash<cudaq::CompileTarget> {
  std::size_t operator()(const cudaq::CompileTarget &t) const noexcept;
};
