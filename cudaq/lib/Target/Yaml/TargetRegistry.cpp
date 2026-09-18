/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetRegistry.h"
#include "TargetConfigHelper.h"
#include "cudaq/Target/TargetDatabase.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include <algorithm>
#include <deque>
#include <iostream>
#include <map>

using namespace cudaq::config;
using namespace cudaq::config::detail;

/// `base` with the host platform's shared library extension appended.
static std::string withSharedLibExt(std::string_view base) {
  return std::string(base) + std::string(kSharedLibraryExtension);
}

static std::string hyphenToUnderscore(std::string name) {
  std::replace(name.begin(), name.end(), '-', '_');
  return name;
}

static bool fileExistsIn(const std::vector<std::filesystem::path> &dirs,
                         const std::string &fileName) {
  for (const auto &dir : dirs) {
    if (std::filesystem::exists(dir / fileName))
      return true;
  }
  return false;
}

static bool isFp64(const cudaq::config::BackendEndConfigEntry &backend) {
  for (const auto &def : backend.PreprocessorDefines)
    if (def.find("CUDAQ_SIMULATION_SCALAR_FP64") != std::string::npos)
      return true;
  for (const auto &sim : backend.SimulationBackend.values)
    if (sim.find("fp64") != std::string::npos)
      return true;
  return false;
}

static std::vector<std::filesystem::path>
searchDirs(const TargetEntry &entry, const HostEnvironment &env) {
  std::vector<std::filesystem::path> dirs = env.libraryPaths;
  if (!entry.pluginLibDir.empty())
    dirs.push_back(entry.pluginLibDir);
  return dirs;
}

bool findPluginLibrary(const std::string &name,
                       const std::vector<std::filesystem::path> &dirs) {
  for (const auto &candidate : sharedLibraryNameCandidates(name)) {
    const std::filesystem::path requested(candidate);
    if (requested.is_absolute()) {
      if (std::filesystem::exists(requested))
        return true;
      continue;
    }
    if (fileExistsIn(dirs, candidate))
      return true;
  }
  return false;
}

cudaq::config::TargetRegistry::TargetRegistry() {
  for (const auto &[name, config] : listBuiltinTargets()) {
    entries.push_back({
        .name = std::string(name),
        .config = config,
        .origin = TargetOrigin::Builtin,
    });
  }
}

bool cudaq::config::TargetRegistry::addEntry(
    const std::string &name, TargetOrigin origin,
    const std::filesystem::path &configPath,
    const std::filesystem::path &pluginLibDir, TargetConfig config) {
  if (lookup(name))
    return false;
  ownedConfigs.push_back(std::move(config));
  entries.push_back({.name = name,
                     .config = &ownedConfigs.back(),
                     .origin = origin,
                     .configPath = configPath,
                     .pluginLibDir = pluginLibDir});
  return true;
}

bool cudaq::config::TargetRegistry::addTargetConfigFile(
    const std::filesystem::path &configPath) {
  TargetConfig config;
  try {
    config = loadTargetConfig(configPath);
  } catch (const std::exception &ex) {
    std::cerr << "warning: skipping target YAML " << configPath.string() << ": "
              << ex.what() << "\n";
    return false;
  }
  return addEntry(configPath.stem().string(), TargetOrigin::YamlFile,
                  configPath, /*pluginLibDir=*/{}, std::move(config));
}

std::vector<std::string> cudaq::config::TargetRegistry::addPluginRoot(
    const std::filesystem::path &root) {
  std::vector<std::string> added;
  const auto targetsDir = root / "targets";
  if (!std::filesystem::is_directory(targetsDir))
    return added;

  const auto pluginLibDir = root / "lib";
  std::vector<std::filesystem::directory_entry> files;
  for (const auto &file : std::filesystem::directory_iterator{targetsDir})
    files.push_back(file);
  std::sort(files.begin(), files.end(), [](const auto &a, const auto &b) {
    return a.path().filename() < b.path().filename();
  });

  // Prefer compiled plugin libraries over YAML of the same stem.
  std::map<std::string, std::filesystem::path> libraryByName;
  std::map<std::string, std::filesystem::path> ymlByName;
  for (const auto &file : files) {
    auto path = file.path();
    auto stem = path.stem().string();
    auto ext = path.extension().string();
    if (isSharedLibraryExtension(ext))
      libraryByName.emplace(stem, path);
    else if (ext == ".yml" || ext == ".yaml")
      ymlByName.emplace(stem, path);
  }

  auto skipOrAdd = [&](const std::string &name, TargetOrigin origin,
                       const std::filesystem::path &configPath,
                       TargetConfig config) {
    const auto libDir = std::filesystem::is_directory(pluginLibDir)
                            ? pluginLibDir
                            : std::filesystem::path{};
    if (!addEntry(name, origin, configPath, libDir, std::move(config))) {
      std::cerr << "warning: skipping target '" << name << "' from "
                << configPath.string()
                << "; a target with that name is already registered\n";
      return;
    }
    added.push_back(name);
  };

  for (const auto &[name, path] : libraryByName) {
    auto loaded = loadTargetPluginLibrary(path);
    if (!loaded.ok) {
      std::cerr << "warning: skipping target plugin library " << path.string()
                << ": " << loaded.error << "\n";
      continue;
    }
    skipOrAdd(name, TargetOrigin::PluginLibrary, path,
              std::move(loaded.config));
  }
  for (const auto &[name, path] : ymlByName) {
    if (libraryByName.count(name))
      continue;
    try {
      auto config = loadTargetConfig(path, root);
      skipOrAdd(name, TargetOrigin::YamlFile, path, std::move(config));
    } catch (const std::exception &ex) {
      std::cerr << "warning: skipping target YAML " << path.string() << ": "
                << ex.what() << "\n";
    }
  }
  return added;
}

const TargetEntry *
cudaq::config::TargetRegistry::lookup(std::string_view name) const {
  for (const auto &entry : entries)
    if (entry.name == name)
      return &entry;
  return nullptr;
}

std::vector<const TargetEntry *> cudaq::config::TargetRegistry::list() const {
  std::vector<const TargetEntry *> result;
  result.reserve(entries.size());
  for (const auto &entry : entries)
    result.push_back(&entry);
  std::sort(result.begin(), result.end(),
            [](const TargetEntry *a, const TargetEntry *b) {
              return a->name < b->name;
            });
  return result;
}

cudaq::config::ResolvedTarget cudaq::config::TargetRegistry::resolveEntry(
    const TargetEntry &entry, const HostEnvironment &env,
    const std::map<std::string, std::string> &args) const {
  ResolvedTarget result;
  result.entry = &entry;
  const auto *backend = selectBackend(*entry.config, args);
  if (backend) {
    result.resolved.platformName =
        backend->PlatformLibrary.empty()
            ? "default"
            : hyphenToUnderscore(backend->PlatformLibrary);
    result.resolved.fp64Simulation = isFp64(*backend);

    auto dirs = searchDirs(entry, env);
    if (!backend->SimulationBackend.values.empty()) {
      bool found = false;
      for (const auto &sim : backend->SimulationBackend.values) {
        const auto libName = withSharedLibExt("libnvqir-" + sim);
        if (fileExistsIn(dirs, libName)) {
          result.resolved.simulatorName = hyphenToUnderscore(sim);
          found = true;
          break;
        }
      }
      if (!found) {
        result.status.availability = Availability::MissingSimulator;
        result.status.diagnostic =
            "Target '" + entry.name +
            "' requires an NVQIR simulator library that was not found.";
        return result;
      }
    }

    if (!backend->PlatformLibrary.empty()) {
      const auto plat = backend->PlatformLibrary;
      const auto libName = withSharedLibExt("libcudaq-platform-" + plat);
      const auto libNameUs =
          withSharedLibExt("libcudaq-platform-" + hyphenToUnderscore(plat));
      if (!fileExistsIn(dirs, libName) && !fileExistsIn(dirs, libNameUs)) {
        result.status.availability = Availability::MissingPlatformLibrary;
        result.status.diagnostic = "Target '" + entry.name +
                                   "' requires platform library '" + plat +
                                   "', which was not found.";
        return result;
      }
    }

    for (const auto &plugin : entry.config->PluginLibraries) {
      if (!findPluginLibrary(plugin, dirs)) {
        result.status.availability = Availability::MissingPluginLibrary;
        result.status.diagnostic = "Target '" + entry.name +
                                   "' requires plugin library '" + plugin +
                                   "', which was not found.";
        return result;
      }
    }
  } else {
    result.resolved.platformName = "default";
  }

  if (entry.config->GpuRequired && env.gpuCount == 0) {
    result.status.availability = Availability::RequiresGpu;
    result.status.diagnostic =
        "Target '" + entry.name +
        "' requires an NVIDIA GPU, but none was detected on this host.";
    return result;
  }

  if (entry.origin != TargetOrigin::Builtin) {
    const auto compatibility = checkExternalTargetVersion(
        *entry.config, env.cudaqVersion, entry.configPath);
    if (compatibility.Status == TargetVersionCompatibility::Error) {
      result.status.availability = Availability::IncompatibleVersion;
      result.status.diagnostic = compatibility.Diagnostic;
      return result;
    }
    if (compatibility.Status == TargetVersionCompatibility::Warning)
      result.status.diagnostic = compatibility.Diagnostic;
  }

  result.status.availability = Availability::Available;
  return result;
}

std::optional<ResolvedTarget> cudaq::config::TargetRegistry::resolve(
    std::string_view name, const HostEnvironment &env,
    const std::map<std::string, std::string> &args) const {
  const auto *entry = lookup(name);
  if (!entry)
    return std::nullopt;
  return resolveEntry(*entry, env, args);
}

std::vector<ResolvedTarget>
cudaq::config::TargetRegistry::resolveAll(const HostEnvironment &env) const {
  std::vector<ResolvedTarget> result;
  for (const auto *entry : list())
    result.push_back(resolveEntry(*entry, env));
  return result;
}

std::string
cudaq::config::emitNvqppConfig(const TargetEntry &entry,
                               const std::map<std::string, std::string> &args) {
  return processRuntimeArgs(*entry.config, args);
}
