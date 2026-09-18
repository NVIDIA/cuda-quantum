/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::config {

/// Filename extension of a shared library on the host platform.
#ifdef __APPLE__
inline constexpr std::string_view kSharedLibraryExtension = ".dylib";
#else
inline constexpr std::string_view kSharedLibraryExtension = ".so";
#endif

/// Every filename extension that denotes a shared library on some platform
/// CUDA-Q supports.
inline constexpr std::array<std::string_view, 2> kSharedLibraryExtensions = {
    ".so", ".dylib"};

/// Whether `extension` (including the leading dot) denotes a shared library.
inline bool isSharedLibraryExtension(std::string_view extension) {
  for (auto candidate : kSharedLibraryExtensions)
    if (extension == candidate)
      return true;
  return false;
}

/// The filenames to try, in order, for a shared library a target
/// configuration refers to as `name`.
inline std::vector<std::string>
sharedLibraryNameCandidates(std::string_view name) {
  std::vector<std::string> candidates{std::string(name)};

  std::string_view stem = name;
  for (auto extension : kSharedLibraryExtensions) {
    if (name.size() > extension.size() && name.ends_with(extension)) {
      stem = name.substr(0, name.size() - extension.size());
      break;
    }
  }

  auto hostName = std::string(stem) + std::string(kSharedLibraryExtension);
  if (hostName != candidates.front())
    candidates.push_back(std::move(hostName));
  return candidates;
}

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

} // namespace cudaq::config
