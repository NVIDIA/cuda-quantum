/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// YAML schema for target configuration and entry points for parsing.

#include "v1/TargetConfigV1.h"

namespace cudaq::config {

/// All supported schema versions are listed here.
///
/// Add new schema versions here and update the `createTargetConfigFromSchema`.
/// Unversioned YAML files are treated as v1.
using TargetConfigSchema = rfl::TaggedUnion<"version", v1::TargetConfigV1>;

/// Convert a parsed schema configuration to the in-memory type.
TargetConfig createTargetConfigFromSchema(const TargetConfigSchema &schema);

/// Parse target configuration YAML from text.
///
/// If a single argument is passed, the text must have already been preprocessed
/// by `substitutePluginRoot`.
///
/// Throws std::runtime_error on malformed YAML, schema violations, or
/// validation failures.
TargetConfig parseTargetConfig(const std::string &substitutedYamlContent);
TargetConfig parseTargetConfig(std::string yamlContent,
                               const std::filesystem::path &pluginRoot);

std::string substitutePluginRoot(std::string yamlContent,
                                 const std::filesystem::path &pluginRoot);

} // namespace cudaq::config
