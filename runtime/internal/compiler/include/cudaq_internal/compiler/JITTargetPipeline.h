/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <map>
#include <string>

namespace cudaq::config {
class TargetConfig;
}

namespace cudaq_internal::compiler {

/// Target JIT pipeline settings resolved from target data and runtime options.
struct JITTargetPipelineConfig {
  /// Build the staged JIT target pipeline or exact target override.
  static JITTargetPipelineConfig createFromTargetConfig(
      const cudaq::config::TargetConfig &config,
      const std::map<std::string, std::string> &runtimeConfig, bool emulate);

  /// MLIR pass pipeline to run.
  std::string passPipelineConfig = "canonicalize";
  /// Code generation emission selected by the target.
  std::string codegenTranslation;
  /// Optional pass pipeline to run after code generation.
  std::string postCodeGenPasses;
  /// Target has a top-level backend entry.
  bool hasBackendConfig = false;
  /// Backend data provides target pass pipeline fields.
  bool hasConfiguredPassPipeline = false;
  /// `target-pass-pipeline` is an exact pipeline override.
  bool usesTargetPassPipelineOverride = false;
  /// Standard `jit-finalize-pipeline` is part of `passPipelineConfig`.
  bool runsStandardFinalize = false;
};

/// Replace `%KEY%` and `%KEY:default%` placeholders from runtime options.
void substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig);

/// Rewrite `qubit-mapping{...device=...}` to use `device=bypass`.
void setQubitMappingBypass(std::string &pipeline);

} // namespace cudaq_internal::compiler
