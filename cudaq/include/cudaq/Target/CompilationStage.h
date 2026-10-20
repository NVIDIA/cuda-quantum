/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/Profile/AllProfiles.h"
#include <string>
#include <variant>

namespace cudaq::config {

/// A lowering that prescribes an explicit MLIR pass pipeline for the
/// compilation stage.
struct CustomPipelineLowering {
  /// If set, override the whole compilation pipeline with this string.
  std::string overridePassPipeline;

  /// Compilation pipeline to insert at the appropriate stages.
  std::string highLevelPipeline;
  std::string midLevelPipeline;
  std::string lowLevelPipeline;

  /// Whether the pipeline is empty.
  bool empty() const {
    return overridePassPipeline.empty() && highLevelPipeline.empty() &&
           midLevelPipeline.empty() && lowLevelPipeline.empty();
  }

  bool operator==(const CustomPipelineLowering &other) const = default;
};

/// All supported lowering kinds. Add handler-based lowerings here.
using Lowering = std::variant<CustomPipelineLowering>;

/// One stage of the compilation pipeline: the IR profile the stage lowers to
/// and how to get there.
struct CompilationStage {
  /// The IR profile the source program must satisfy after this stage.
  Profile irOutputProfile;

  /// How to lower the input IR to `irOutputProfile`.
  Lowering lowering;

  bool operator==(const CompilationStage &other) const = default;
};

/// Code generation specification.
struct Codegen {
  std::string emissionFormat;
  std::string postCodegenPasses;

  bool operator==(const Codegen &other) const = default;
};

} // namespace cudaq::config
