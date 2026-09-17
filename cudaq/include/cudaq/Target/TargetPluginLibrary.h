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
#include <string>

namespace cudaq::config {

/// The exported C symbol name a compiled external target plugin library must
/// define. Consumers will `dlopen()` the library and `dlsym()` exactly this
/// name.
///
/// This name is the load-time equivalent of a link-time ABI version check: a
/// plugin library built against a different `TargetConfig`/generator ABI
/// exports (or is looked up under) a different symbol name entirely, so a
/// mismatch fails immediately and unambiguously at `dlsym` time.
inline constexpr const char *kTargetPluginSymbolName = "cudaq_target_config_v1";

/// Signature of the symbol named by `kTargetPluginSymbolName`.
using TargetPluginEntryPoint = const TargetConfig *(*)();

/// Result of attempting to load a compiled external target plugin library.
struct TargetPluginLoadResult {
  /// Non-null on success; always heap-owned by the caller via `config` below
  /// so the shared library may be safely left loaded/unloaded either way -
  /// the returned `TargetConfig` is a copy, not a pointer into the library.
  bool ok = false;
  TargetConfig config;
  /// Populated only when `ok` is false: a human-readable diagnostic safe to
  /// print directly.
  std::string error;
};

/// `dlopen()`s `libraryPath`, resolves `kTargetPluginSymbolName`, invokes it,
/// and copies out the resulting `TargetConfig`. This is the *only* supported
/// way to load an external/plugin target as of the switch away from raw,
/// unvalidated `.yml` text files placed under a plugin's `targets/`
/// directory.
TargetPluginLoadResult
loadTargetPluginLibrary(const std::filesystem::path &libraryPath);

} // namespace cudaq::config
