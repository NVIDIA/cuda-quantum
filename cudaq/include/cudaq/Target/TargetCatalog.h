/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
#include <deque>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::config {

namespace detail {
enum class TargetOrigin { Builtin, PluginLibrary, YamlFile };

enum class Availability {
  Available,
  RequiresGpu,
  MissingSimulator,
  MissingPlatformLibrary,
  MissingPluginLibrary,
  IncompatibleVersion
};

/// Host-independent discovery facts about one target.
struct TargetEntry {
  std::string name;
  const TargetConfig *config = nullptr;
  TargetOrigin origin = TargetOrigin::Builtin;
  std::filesystem::path configPath = {};
  std::filesystem::path pluginLibDir = {};
};
} // namespace detail

struct ResolvedTarget {
  struct ResolvedArtifacts {
    std::string simulatorName;
    std::string platformName = "default";
    bool fp64Simulation = false;
  };

  struct TargetStatus {
    detail::Availability availability = detail::Availability::Available;
    std::string diagnostic;
    bool isAvailable() const {
      return availability == detail::Availability::Available;
    }
  };

  const detail::TargetEntry *entry = nullptr;
  ResolvedArtifacts resolved;
  TargetStatus status;
};

/// Host facts, supplied per query so the library holds no global state and
/// needs no CUDA or runtime dependency.
struct HostEnvironment {
  unsigned gpuCount = 0;
  std::string cudaqVersion;
  std::vector<std::filesystem::path> libraryPaths;
};

class TargetCatalog {
  using TargetEntry = detail::TargetEntry;

public:
  /// Seeded from the linked-in built-in target database.
  TargetCatalog();
  TargetCatalog(const TargetCatalog &) = delete;
  TargetCatalog &operator=(const TargetCatalog &) = delete;
  TargetCatalog(TargetCatalog &&) = default;
  TargetCatalog &operator=(TargetCatalog &&) = default;

  /// Scan `<root>/targets/` for compiled target plugin libraries (preferred)
  /// and `*.yml` files. Names that already exist (built-in or previously added)
  /// are skipped with a diagnostic on `stderr`. Returns the names that were
  /// added.
  std::vector<std::string> addPluginRoot(const std::filesystem::path &root);

  /// Register the single target configuration at `configPath` under its
  /// filename stem. `configPath` may be a YAML file or a compiled target plugin
  /// library. Unlike `addPluginRoot` this expects no plugin directory layout:
  /// the file stands alone, so there is no sibling `lib/` to search and the
  /// target may only name libraries already on the CUDA-Q library path. Returns
  /// false if the name is already registered or the file could not be loaded.
  bool addTargetConfigFile(const std::filesystem::path &configPath);

  const TargetEntry *lookup(std::string_view name) const;
  std::vector<const TargetEntry *> list() const;

  std::optional<ResolvedTarget>
  resolve(std::string_view name, const HostEnvironment &env,
          const std::map<std::string, std::string> &args = {}) const;
  std::vector<ResolvedTarget> resolveAll(const HostEnvironment &env) const;

private:
  std::vector<TargetEntry> entries;
  std::deque<TargetConfig> ownedConfigs;
  /// Take ownership of `config` and register it under `name`. Returns false
  /// without registering anything if `name` is already taken.
  bool addEntry(const std::string &name, detail::TargetOrigin origin,
                const std::filesystem::path &configPath,
                const std::filesystem::path &pluginLibDir, TargetConfig config);
  ResolvedTarget
  resolveEntry(const TargetEntry &entry, const HostEnvironment &env,
               const std::map<std::string, std::string> &args = {}) const;
};

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

/// Emit the bash `KEY=value` assignments consumed by `nvq++`.
std::string emitNvqppConfig(const detail::TargetEntry &entry,
                            const std::map<std::string, std::string> &args);

} // namespace cudaq::config
