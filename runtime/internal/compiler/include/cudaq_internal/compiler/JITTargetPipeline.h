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

struct JITTargetPipelineConfig {
  std::string passPipelineConfig = "canonicalize";
  std::string codegenTranslation;
  std::string postCodeGenPasses;
  bool hasBackendConfig = false;
  bool hasConfiguredPassPipeline = false;
  bool usesTargetPassPipelineOverride = false;
  bool runsStandardFinalize = false;
};

JITTargetPipelineConfig buildJITTargetPipelineConfig(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &runtimeConfig, bool emulate);

void substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig);

void setQubitMappingBypass(std::string &pipeline);

} // namespace cudaq_internal::compiler
