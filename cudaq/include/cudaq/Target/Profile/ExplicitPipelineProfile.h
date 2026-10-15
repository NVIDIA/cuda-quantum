/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>

namespace cudaq::config {

/// @brief IR profile prescribing an explicit MLIR pass pipeline.
///
/// This is a transitional profile type. Unlike declarative IR profiles, it
/// does not describe which IR is legal for the target; it directly prescribes
/// the compilation pipeline to execute.
struct ExplicitPipelineProfile {
  /// If set, override the whole compilation pipeline with this string.
  std::string overridePassPipeline;

  /// Compilation pipeline to insert at the appropriate stages.
  std::string highLevelPipeline;
  std::string midLevelPipeline;
  std::string lowLevelPipeline;

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

  bool operator==(const ExplicitPipelineProfile &other) const = default;
};

} // namespace cudaq::config
