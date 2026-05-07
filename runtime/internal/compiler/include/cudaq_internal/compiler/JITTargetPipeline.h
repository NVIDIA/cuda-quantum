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

/// Materialized target JIT pipeline settings derived from a target YAML config
/// and runtime target arguments.
struct JITTargetPipelineConfig {
  /// MLIR pass-pipeline string to run. Defaults to the compiler's baseline
  /// canonicalization-only pipeline when the target has no backend config.
  std::string passPipelineConfig = "canonicalize";
  /// Codegen emission selected by the target config and runtime args, e.g.
  /// "qir-adaptive", "qir-base", "qasm2", or "nop".
  std::string codegenTranslation;
  /// Optional pass-pipeline string to run after codegen translation.
  std::string postCodeGenPasses;
  /// True when the target has a top-level backend `config:` entry.
  bool hasBackendConfig = false;
  /// True when the backend config provides any target pass-pipeline fields.
  /// Python local targets use this to avoid adding default JIT stages to
  /// simulator targets that never requested target pipeline handling.
  bool hasConfiguredPassPipeline = false;
  /// True when `target-pass-pipeline` is set. In this mode the target-provided
  /// pipeline is used exactly, with no standard prep/deploy/finalize stages
  /// interleaved.
  bool usesTargetPassPipelineOverride = false;
  /// True when `passPipelineConfig` includes the standard
  /// `jit-finalize-pipeline`. Python translate paths use this to avoid running
  /// target finalization twice.
  bool runsStandardFinalize = false;
};

/// Build the standard staged JIT target pipeline:
///
///   canonicalize, {hw|emul}-jit-prep-pipeline,
///   jit-high-level-pipeline, jit-deploy-pipeline,
///   jit-mid-level-pipeline, jit-finalize-pipeline,
///   jit-low-level-pipeline
///
/// `target-pass-pipeline` remains an exact override. `%KEY%` and
/// `%KEY:default%` placeholders are substituted from `runtimeConfig`.
JITTargetPipelineConfig buildJITTargetPipelineConfig(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &runtimeConfig, bool emulate);

/// Replace `%KEY%` and `%KEY:default%` placeholders in `pipeline` using
/// lowercase keys from `runtimeConfig`. Unresolved placeholders without a
/// default are left unchanged for provider-specific rewriting.
void substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig);

/// Rewrite any `qubit-mapping{...device=...}` stage in `pipeline` to
/// `device=bypass`, preserving the other pass options.
void setQubitMappingBypass(std::string &pipeline);

} // namespace cudaq_internal::compiler
