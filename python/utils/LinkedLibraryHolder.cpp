/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "common/FmtCore.h"
#include "common/PluginUtils.h"
#include "nvqir/CircuitSimulator.h"
#include "cudaq/Support/Plugin.h"
#include "cudaq/Target/TargetRegistry.h"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/target_control.h"
#include <cstdlib>
#include <regex>
#include <sstream>
#include <string>

// Our hooks into configuring the NVQIR backend.
extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *);
void __nvqir__setSimulatorInitCallback(void (*)());
}

// Our hook into configuring the quantum platform.
extern "C" void setQuantumPlatformInitCallback(void (*)());

namespace cudaq::mpi {
void set_communicator(void *comm);
}

namespace cudaq {

int num_available_gpus();

void setQuantumPlatformInternal(quantum_platform *p);
void setExecutionManagerInternal(ExecutionManager *em);
void resetExecutionManagerInternal();

// File-scoped pointer for the NVQIR/platform lazy init callbacks.
static LinkedLibraryHolder *activeHolder = nullptr;
static void lazyInitSimulator() {
  if (activeHolder && !activeHolder->isTargetInitialized())
    activeHolder->resetTarget();
}

static bool isSimulationConfig(const cudaq::config::TargetConfig &config) {
  const cudaq::config::BackendEndConfigEntry *backend = nullptr;
  for (const auto &entry : config.ConfigMap)
    if (entry.Default.has_value() && entry.Default.value())
      backend = &entry.Config;
  if (!backend && config.BackendConfig)
    backend = &*config.BackendConfig;
  if (!backend)
    return false;
  return backend->PlatformQpu.empty() &&
         backend->LibraryModeExecutionManager.empty();
}

static void addPluginScopeToRegistry(cudaq::config::TargetRegistry &registry,
                                     const std::filesystem::path &scope) {
  if (!std::filesystem::is_directory(scope))
    return;
  for (const auto &entry : std::filesystem::directory_iterator{scope}) {
    if (entry.is_directory())
      registry.addPluginRoot(entry.path());
  }
}

static std::filesystem::path userPluginScope() {
  if (const char *pluginRoot = std::getenv("CUDAQ_PLUGIN_ROOT"))
    return pluginRoot;
  if (const char *xdg = std::getenv("XDG_DATA_HOME"))
    return std::filesystem::path(xdg) / "cudaq" / "plugins";
  if (const char *home = std::getenv("HOME"))
    return std::filesystem::path(home) / ".local" / "share" / "cudaq" /
           "plugins";
  return {};
}

cudaq::config::HostEnvironment LinkedLibraryHolder::hostEnv() const {
  return cudaq::detail::currentHostEnvironment();
}

void LinkedLibraryHolder::addPluginScope(const std::filesystem::path &scope) {
  addPluginScopeToRegistry(targetRegistry, scope);
}

RuntimeTarget LinkedLibraryHolder::makeRuntimeTarget(
    const config::ResolvedTarget &resolved) const {
  RuntimeTarget target;
  target.name = resolved.entry->name;
  target.config = *resolved.entry->config;
  target.description = resolved.entry->config->Description;
  target.simulatorName = resolved.resolved.simulatorName;
  target.platformName = resolved.resolved.platformName;
  target.precision = resolved.resolved.fp64Simulation
                         ? simulation_precision::fp64
                         : simulation_precision::fp32;
  target.pluginLibDir = resolved.entry->pluginLibDir.string();
  target.configPath = resolved.entry->configPath;
  if (!resolved.status.isAvailable())
    target.availabilityDiagnostic = resolved.status.diagnostic;
  return target;
}

void LinkedLibraryHolder::reloadTargets() {
  targets.clear();
  simulationTargets.clear();
  for (const auto &resolved : targetRegistry.resolveAll(hostEnv())) {
    auto target = makeRuntimeTarget(resolved);
    CUDAQ_INFO("Found Target: {} -> (sim={}, platform={}, available={})",
               target.name, target.simulatorName, target.platformName,
               resolved.status.isAvailable());
    targets.emplace(target.name, target);
    if (isSimulationConfig(*resolved.entry->config) &&
        resolved.status.isAvailable())
      simulationTargets.emplace(target.name, target);
  }
}

LinkedLibraryHolder::LinkedLibraryHolder() : availablePlatforms{"default"} {
  ScopedTraceWithContext("LinkedLibraryHolder::constructor");
  CUDAQ_INFO("Init infrastructure for pythonic builder.");

  if (!cudaq::detail::canModifyTarget())
    return;

  cudaq::detail::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  libSuffix = "dylib";
  cudaq::detail::getCUDAQLibraryPath(&data);
#else
  libSuffix = "so";
  dl_iterate_phdr(cudaq::detail::getCUDAQLibraryPath, &data);
#endif

  std::filesystem::path nvqirLibPath{data.path};
  cudaqLibPath = nvqirLibPath.parent_path();
  if (cudaqLibPath.filename().string() == "common") {
    // this is a build path
    cudaqLibPath = cudaqLibPath.parent_path().parent_path() / "lib";
  }

  // Populate the map of available targets.
  {
    ScopedTraceWithContext("findAvailableTargets");
    addPluginScope(userPluginScope());
    addPluginScope(cudaqLibPath.parent_path() / "plugins");
    reloadTargets();
  }

  CUDAQ_INFO("Init: Library Path is {}.", cudaqLibPath.string());

  // Load nvqir, cudaq, and the default execution manager. The em cannot
  // be a needed dep of libcudaq.so (circular dependency), but downstream
  // libraries like cuda-qx reference its symbols at dlopen time.
  std::vector<std::filesystem::path> libPaths{
      cudaqLibPath / fmt::format("libnvqir.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq-em-default.{}", libSuffix)};

  const char *dynlibs_var = std::getenv("CUDAQ_DYNLIBS");
  if (dynlibs_var != nullptr) {
    std::string dynlib;
    std::stringstream ss((std::string(dynlibs_var)));
    while (std::getline(ss, dynlib, ':')) {
      CUDAQ_INFO("Init: add dynamic library path {}.", dynlib);
      libPaths.push_back(dynlib);
    }
  }

  // Load all the defaults
  {
    ScopedTraceWithContext("dlopen_core_and_dynlibs");
    for (auto &p : libPaths) {
      void *libHandle = dlopen(p.string().c_str(), RTLD_GLOBAL | RTLD_NOW);
      if (libHandle) {
        libHandles.emplace(p.string(), libHandle);
      } else {
        char *error_msg = dlerror();
        CUDAQ_INFO("Failed to load '{}': ERROR '{}'", p.string(),
                   (error_msg ? std::string(error_msg) : "unknown."));
      }
    }
  } // end dlopen_core_and_dynlibs

  // directory_iterator ordering is unspecified, so sort it to make it
  // repeatable and consistent.
  std::vector<std::filesystem::directory_entry> entries;
  for (const auto &entry : std::filesystem::directory_iterator{cudaqLibPath})
    entries.push_back(entry);
  std::sort(entries.begin(), entries.end(),
            [](const std::filesystem::directory_entry &a,
               const std::filesystem::directory_entry &b) {
              return a.path().filename() < b.path().filename();
            });

  {
    ScopedTraceWithContext("scan_simulator_filenames");
    // Discover available simulators and platforms by scanning filenames.
    // Libraries are loaded on demand in getSimulator()/getPlatform() rather
    // than eagerly here, to avoid the cost of dlopen'ing all .so files at
    // import time.
    for (const auto &library : entries) {
      auto path = library.path();
      auto fileName = path.filename().string();
      if (fileName.find("nvqir-") != std::string::npos) {
        auto simName =
            std::regex_replace(fileName, std::regex("libnvqir-"), "");
        simName = std::regex_replace(simName, std::regex("-"), "_");
        auto idx = simName.find_last_of(".");
        simName = simName.substr(0, idx);
        simulatorLibPaths.emplace(simName, path);
        availableSimulators.push_back(simName);
        CUDAQ_INFO("Found simulator plugin {}.", simName);
      } else if (fileName.find("cudaq-platform-") != std::string::npos) {
        auto platformName =
            std::regex_replace(fileName, std::regex("libcudaq-platform-"), "");
        platformName = std::regex_replace(platformName, std::regex("-"), "_");
        auto idx = platformName.find_last_of(".");
        platformName = platformName.substr(0, idx);
        platformLibPaths.emplace(platformName, path);
        availablePlatforms.push_back(platformName);
        CUDAQ_INFO("Found platform plugin {}.", platformName);
      }
    }
  } // end scan_simulator_filenames

  // Capture CUDAQ_DEFAULT_SIMULATOR now so it is visible when
  // resolveDefaultTarget() runs later during deferred initialization.
  // This must happen at import time because the env var may be set
  // programmatically before import (e.g., in test files).
  auto envSim = std::getenv("CUDAQ_DEFAULT_SIMULATOR");
  if (envSim)
    cachedDefaultSimulatorEnv = envSim;

  // Default to qpp-cpu. The full target resolution (GPU detection, simulator
  // loading) is deferred to first use via the NVQIR callback or getTarget().
  defaultTarget = "qpp-cpu";
  currentTarget = defaultTarget;
  activeHolder = this;
  __nvqir__setSimulatorInitCallback(lazyInitSimulator);
  setQuantumPlatformInitCallback(lazyInitSimulator);
}

LinkedLibraryHolder::~LinkedLibraryHolder() {
  activeHolder = nullptr;
  __nvqir__setSimulatorInitCallback(nullptr);
  setQuantumPlatformInitCallback(nullptr);
  for (auto &[name, handle] : libHandles) {
    if (handle)
      dlclose(handle);
  }
}

/// @brief Ensure a library is loaded, dlopen'ing it on demand if needed.
void LinkedLibraryHolder::ensureLibLoaded(const std::filesystem::path &path) {
  auto pathStr = path.string();
  if (libHandles.count(pathStr))
    return;
  void *handle = dlopen(pathStr.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (!handle) {
    char *error_msg = dlerror();
    throw std::runtime_error(
        fmt::format("Failed to load library '{}': {}", pathStr,
                    (error_msg ? std::string(error_msg) : "unknown")));
  }
  libHandles.emplace(pathStr, handle);

  // If the library exports cudaqGetPluginInfo, call it to register any MLIR
  // passes or other extensions provided by the plugin.
  using PluginInfoFn = cudaq::PluginLibraryInfo (*)();
  auto *getInfoFn =
      reinterpret_cast<PluginInfoFn>(dlsym(handle, "cudaqGetPluginInfo"));
  if (getInfoFn) {
    auto info = getInfoFn();
    if (info.RegisterCallbacks) {
      CUDAQ_INFO("Registering MLIR extensions from plugin '{}'.",
                 info.pluginName ? info.pluginName : pathStr.c_str());
      info.RegisterCallbacks();
    }
  }
}

nvqir::CircuitSimulator *
LinkedLibraryHolder::getSimulator(const std::string &simName) {
  auto end = availableSimulators.end();
  auto iter = std::find(availableSimulators.begin(), end, simName);
  if (iter == end)
    throw std::runtime_error("Invalid simulator requested: " + simName);

  // Ensure the simulator library is loaded on demand. Since ensureLibLoaded
  // uses RTLD_GLOBAL, the symbols are globally visible and
  // getUniquePluginInstance can find them via dlopen(nullptr).
  auto pathIter = simulatorLibPaths.find(simName);
  if (pathIter != simulatorLibPaths.end())
    ensureLibLoaded(pathIter->second);

  return getUniquePluginInstance<nvqir::CircuitSimulator>(
      std::string("getCircuitSimulator_") + simName);
}

quantum_platform *
LinkedLibraryHolder::getPlatform(const std::string &platformName) {
  auto end = availablePlatforms.end();
  auto iter = std::find(availablePlatforms.begin(), end, platformName);
  if (iter == end)
    throw std::runtime_error("Invalid platform requested: " + platformName);

  auto pathIter = platformLibPaths.find(platformName);
  if (pathIter != platformLibPaths.end())
    ensureLibLoaded(pathIter->second);

  return getUniquePluginInstance<quantum_platform>(
      std::string("getQuantumPlatform_") + platformName);
}

/// @brief Determine the best default target based on GPU availability and
/// installed simulators. No simulator dlopen, but the first call triggers
/// CUDA driver init via `num_available_gpus()` for GPU detection.
std::string LinkedLibraryHolder::resolveDefaultTarget() {
  ScopedTraceWithContext("resolveDefaultTarget");
  std::string resolved = "qpp-cpu";

  if (num_available_gpus() > 0) {
    auto iter = targets.find("nvidia");
    if (iter == targets.end()) {
      CUDAQ_INFO("GPU(s) found but nvidia target not found.");
    } else if (iter->second.isAvailable() &&
               simulatorLibPaths.count(iter->second.simulatorName)) {
      resolved = "nvidia";
    } else {
      CUDAQ_INFO("GPU(s) found but simulator '{}' not available.",
                 iter->second.simulatorName);
    }
  }

  // Check env var: use the cached value from import time if available,
  // otherwise read live (for C++ callers that don't go through the
  // constructor).
  auto env = cachedDefaultSimulatorEnv.empty()
                 ? std::getenv("CUDAQ_DEFAULT_SIMULATOR")
                 : cachedDefaultSimulatorEnv.c_str();
  if (env) {
    CUDAQ_INFO("'CUDAQ_DEFAULT_SIMULATOR' = {}", env);
    auto iter = simulationTargets.find(env);
    if (iter != simulationTargets.end())
      resolved = iter->second.name;
  }

  return resolved;
}

void LinkedLibraryHolder::registerBackendPath(
    const std::filesystem::path &pkgRoot) {
  if (!std::filesystem::exists(pkgRoot))
    throw std::runtime_error(
        "register_backend_path: directory does not exist: " + pkgRoot.string());
  if (!std::filesystem::is_directory(pkgRoot))
    throw std::runtime_error("register_backend_path: not a directory: " +
                             pkgRoot.string());
  auto targetPath = pkgRoot / "targets";
  if (!std::filesystem::is_directory(targetPath))
    throw std::runtime_error(
        "register_backend_path: missing 'targets/' subdirectory under " +
        pkgRoot.string());
  CUDAQ_INFO("register_backend_path: loading external backends from '{}'.",
             pkgRoot.string());
  targetRegistry.addPluginRoot(pkgRoot);
  reloadTargets();
}

void LinkedLibraryHolder::resetTarget() {
  defaultTarget = resolveDefaultTarget();
  currentTarget = defaultTarget;
  try {
    setTarget(defaultTarget);
  } catch (const std::runtime_error &e) {
    if (defaultTarget != "qpp-cpu") {
      CUDAQ_INFO("Failed to activate default target '{}': {}. "
                 "Falling back to qpp-cpu.",
                 defaultTarget, e.what());
      defaultTarget = "qpp-cpu";
      currentTarget = defaultTarget;
      setTarget(defaultTarget);
    } else {
      throw;
    }
  }
}

RuntimeTarget LinkedLibraryHolder::getTarget(const std::string &name) {
  if (!targetInitialized)
    resetTarget();
  auto iter = targets.find(name);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + name + ").");
  if (!iter->second.isAvailable())
    throw std::runtime_error(iter->second.availabilityDiagnostic);

  return iter->second;
}

RuntimeTarget LinkedLibraryHolder::getTarget() {
  if (!targetInitialized)
    resetTarget();
  auto iter = targets.find(currentTarget);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + currentTarget + ").");
  return iter->second;
}

bool LinkedLibraryHolder::hasTarget(const std::string &name,
                                    bool includeUnavailable) {
  auto iter = targets.find(name);
  if (iter == targets.end())
    return false;
  if (includeUnavailable)
    return true;
  return iter->second.isAvailable();
}

void LinkedLibraryHolder::setTarget(
    const std::string &targetName,
    std::map<std::string, std::string> extraConfig) {
  // Do not set the default target if the disallow
  // flag has been set.
  if (!cudaq::detail::canModifyTarget())
    return;

  auto iter = targets.find(targetName);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + targetName + ").");

  auto resolved = targetRegistry.resolve(targetName, hostEnv(), extraConfig);
  if (!resolved)
    throw std::runtime_error("Invalid target name (" + targetName + ").");
  if (!resolved->status.isAvailable())
    throw std::runtime_error(resolved->status.diagnostic);

  auto &target = iter->second;
  target = makeRuntimeTarget(*resolved);
  if (!resolved->status.diagnostic.empty())
    fmt::print(stderr, "{}\n", resolved->status.diagnostic);

  if (!target.config.WarningMsg.empty()) {
    fmt::print(fmt::fg(fmt::color::red), "[warning] ");
    // Output the warning message if any
    fmt::print(fmt::fg(fmt::color::blue), "Target {}: {}\n", target.name,
               target.config.WarningMsg);
  }

  if (!target.config.PluginLibraries.empty()) {
    const auto pythonCAPIName = fmt::format("libcudaqMLIRCAPI.{}", libSuffix);
    std::vector<std::filesystem::path> pythonCAPICandidates{
        cudaqLibPath.parent_path() / "cudaq" / "mlir" / "_mlir_libs" /
            pythonCAPIName,
        cudaqLibPath.parent_path() / "python" / "cudaq" / "mlir" /
            "_mlir_libs" / pythonCAPIName};
    for (const auto &candidatePath : pythonCAPICandidates) {
      if (!std::filesystem::exists(candidatePath))
        continue;

      CUDAQ_INFO("Loading CUDA-Q Python CAPI '{}' for plugin MLIR symbols.",
                 candidatePath.string());
      ensureLibLoaded(candidatePath);
      break;
    }
  }

  auto targetConfigPath = target.configPath;
  cudaq::detail::loadTargetPluginLibraries(targetName, targetConfigPath,
                                           target.config);

  CUDAQ_INFO("Setting target={} (sim={}, platform={})", targetName,
             target.simulatorName, target.platformName);
  std::string simName = target.simulatorName;
  if (simName.empty()) {
    // This target doesn't have a simulator defined, e.g., hardware targets.
    // We still need a simulator in case of local emulation.
    // Ensure defaultTarget is fully resolved (it may still be the initial
    // "qpp-cpu" if deferred initialization hasn't run yet).
    defaultTarget = resolveDefaultTarget();
    auto &defaultTargetInfo = targets[defaultTarget];
    simName = defaultTargetInfo.simulatorName;

    // The precision should match the underlying local simulator that we
    // selected.
    target.precision = defaultTargetInfo.precision;

    // This is really a user error: e.g., using `CUDAQ_DEFAULT_SIMULATOR`
    // environment variable (meant for simulator) to change the default target
    // to some other targets that are not a simulator.
    if (simName.empty())
      throw std::runtime_error("Default target " + defaultTarget +
                               " doesn't define a simulator. Please check your "
                               "CUDAQ_DEFAULT_SIMULATOR environment variable.");
  }
  __nvqir__setCircuitSimulator(getSimulator(simName));
  auto *platform = getPlatform(target.platformName);

  // Provide the already-parsed target config so that
  // DefaultQuantumPlatform::setTargetBackend can skip re-reading the YAML.
  platform->runtimeTarget = std::make_unique<cudaq::RuntimeTarget>(target);

  // Pack the config into the backend string name
  std::string backendConfigStr = targetName;
  for (auto &[key, value] : extraConfig)
    backendConfigStr += fmt::format(";{};{}", key, value);

  if (!target.configPath.empty())
    backendConfigStr +=
        fmt::format(";__target_config_path;{}", target.configPath.string());

  platform->setTargetBackend(backendConfigStr);
  setQuantumPlatformInternal(platform);
  currentTarget = targetName;

  if ("orca-photonics" == targetName) {
    std::filesystem::path libPath =
        cudaqLibPath / fmt::format("libcudaq-em-photonics.{}", libSuffix);
    auto *em = getUniquePluginInstance<ExecutionManager>(
        "getRegisteredExecutionManager_photonics", libPath.c_str());
    setExecutionManagerInternal(em);
  } else {
    resetExecutionManagerInternal();
  }

  // If the config (kwargs) contains comm_handle, set it.
  if (extraConfig.contains("comm_handle")) {
    intptr_t commPtr = std::stoll(extraConfig["comm_handle"]);
    CUDAQ_INFO("Setting communicator for target {} with pointer value {}",
               targetName, commPtr);
    cudaq::mpi::set_communicator(reinterpret_cast<void *>(commPtr));
  }

  targetInitialized = true;
  // Deregister lazy init callbacks now that a target is configured.
  __nvqir__setSimulatorInitCallback(nullptr);
  setQuantumPlatformInitCallback(nullptr);
}

std::vector<RuntimeTarget>
LinkedLibraryHolder::getTargets(bool includeUnavailable) {
  std::vector<RuntimeTarget> ret;
  for (auto &[name, target] : targets) {
    if (!includeUnavailable && !target.isAvailable())
      continue;
    ret.emplace_back(target);
  }
  return ret;
}

std::string python::getTransportLayer(LinkedLibraryHolder *holder) {
  if (holder && cudaq::detail::canModifyTarget()) {
    auto runtimeTarget = holder->getTarget();
    const std::string codegenEmission =
        runtimeTarget.config.getCodeGenSpec(runtimeTarget.runtimeConfig);
    if (!codegenEmission.empty())
      return codegenEmission;
  }
  // Default is full QIR.
  return "qir:0.1";
}
} // namespace cudaq
