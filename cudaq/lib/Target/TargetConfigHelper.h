/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include <filesystem>
#include <map>
#include <string>

namespace cudaq::config {

std::string processRuntimeArgs(const TargetConfig &config,
                               const std::map<std::string, std::string> &args);

/// Select the backend entry implied by `args` (configuration-matrix /
/// default `config:`). Returns nullptr when the target has no backend config.
const BackendEndConfigEntry *
selectBackend(const TargetConfig &config,
              const std::map<std::string, std::string> &args);

/// True after `disableYAMLTargetConfigParsing()` has been called in this
/// process.
bool isDisabledYAMLParsing();

struct TargetPluginLoadResult {
  bool ok = false;
  TargetConfig config;
  std::string error;
};

/// `dlopen()`s `libraryPath`, resolves `kTargetPluginSymbolName`, invokes it,
/// and copies out the resulting `TargetConfig`. When `libraryPath` is
/// `nullptr`, the symbol is looked up in the running program (used by
/// `nvq++ --build-target-from-config`, which links the generated plugin
/// object into the user binary).
TargetPluginLoadResult
loadTargetPluginLibrary(const std::filesystem::path *libraryPath);

} // namespace cudaq::config
