/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/runtime/logger/logger.h"
#include <cctype>

/// Replace `%KEY%` and `%KEY:default%` placeholders from runtime options.
static void substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig) {
  std::string::size_type pos = 0;
  while (pos < pipeline.size()) {
    auto start = pipeline.find('%', pos);
    if (start == std::string::npos)
      break;
    auto end = pipeline.find('%', start + 1);
    if (end == std::string::npos)
      break;
    auto token = pipeline.substr(start + 1, end - start - 1);
    auto colon = token.find(':');
    auto key = (colon != std::string::npos) ? token.substr(0, colon) : token;

    std::string lower;
    for (char c : key)
      lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto it = runtimeConfig.find(lower);

    if (it != runtimeConfig.end()) {
      pipeline.replace(start, end - start + 1, it->second);
      pos = start + it->second.size();
    } else if (colon != std::string::npos) {
      auto defaultVal = token.substr(colon + 1);
      pipeline.replace(start, end - start + 1, defaultVal);
      pos = start + defaultVal.size();
    } else {
      pos = end + 1;
    }
  }
}

cudaq::CompileTarget::CompileTarget(
    config::TargetConfig targetConfig,
    std::map<std::string, std::string> runtimeConfig, bool emulate_)
    : emulate(emulate_) {
  if (!targetConfig.BackendConfig.has_value())
    return;

  const auto &backendConfig = *targetConfig.BackendConfig;
  if (!backendConfig.hasPassPipeline()) {
    // TODO: this means that Remote QPUs with no customisation at all will not
    // run ANY passes. Is this okay?
    pipelineConfig.skipTargetLoweringPipeline = true;
  }

  auto prepPipeline = [&](const std::string &stage,
                          const std::string &stageName) {
    std::string pipeline = stage;
    if (!pipeline.empty()) {
      substitutePipelinePlaceholders(pipeline, runtimeConfig);
      CUDAQ_INFO("{:<27} {}", stageName + ":", pipeline);
    }
    return pipeline;
  };

  if (!backendConfig.TargetPassPipeline.empty()) {
    pipelineConfig.overridePassPipeline = prepPipeline(
        backendConfig.TargetPassPipeline, "Pass pipeline (overridden)");
  } else {
    pipelineConfig.highLevelPipeline =
        prepPipeline(backendConfig.JITHighLevelPipeline, "JIT high level");
    pipelineConfig.midLevelPipeline =
        prepPipeline(backendConfig.JITMidLevelPipeline, "JIT mid level");
    pipelineConfig.lowLevelPipeline =
        prepPipeline(backendConfig.JITLowLevelPipeline, "JIT low level");
  }
  auto codegenTranslation = targetConfig.getCodeGenSpec(runtimeConfig);
  if (!codegenTranslation.empty()) {
    pipelineConfig.codegenTranslation = codegenTranslation;
    CUDAQ_INFO("{:<27} {}\n", "Codegen:", codegenTranslation);
  }
  if (!backendConfig.PostCodeGenPasses.empty()) {
    pipelineConfig.postCodeGenPasses = backendConfig.PostCodeGenPasses;
    CUDAQ_INFO("{:<27} {}\n",
               "Post-codegen:", pipelineConfig.postCodeGenPasses);
  }

  // Handle disable_qubit_mapping runtime option.
  auto disableQM = runtimeConfig.find("disable_qubit_mapping");
  if (disableQM != runtimeConfig.end() && disableQM->second == "true") {
    pipelineConfig.disableQubitMapping = true;
    CUDAQ_INFO("{:<27} {}\n", "disable_qubit_mapping:", "true");
  }
}
