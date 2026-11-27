/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Runtime loading of target configurations from YAML files.

#include "cudaq/Target/TargetConfig.h"
#include <filesystem>

namespace cudaq::config {

/// Disable loading target configurations from YAML for this process. Used by
/// `nvq++`-produced executables to reject any unprocessed YAML files at
/// runtime.
void disableYAMLTargetConfigParsing();

/// Read the target configuration YAML at `configPath` and parse it.
///
/// When `pluginRoot` is empty, `%PLUGIN_ROOT%` resolves against the grandparent
/// of `configPath` (i.e. the plugin root containing `targets/`).
///
/// Throws if YAML parsing has been disabled via
/// `disableYAMLTargetConfigParsing()`.
TargetConfig loadTargetConfig(const std::filesystem::path &configPath,
                              const std::filesystem::path &pluginRoot = {});

} // namespace cudaq::config
