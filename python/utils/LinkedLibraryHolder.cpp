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
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/target_control.h"
#include "nvqir/CircuitSimulator.h"
#include <fstream>
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

// File-scoped pointer for the NVQIR/platform lazy init callbacks.
static LinkedLibraryHolder *activeHolder = nullptr;
static void lazyInitSimulator() {
  if (activeHolder && !activeHolder->isTargetInitialized())
    activeHolder->resetTarget();
}
void setQuantumPlatformInternal(quantum_platform *p);

void setExecutionManagerInternal(ExecutionManager *em);
void resetExecutionManagerInternal();

static constexpr const char PLATFORM_LIBRARY[] = "PLATFORM_LIBRARY=";
static constexpr const char NVQIR_SIMULATION_BACKEND[] =
    "NVQIR_SIMULATION_BACKEND=";
static constexpr const char IS_FP64_SIMULATION[] =
    "CUDAQ_SIMULATION_SCALAR_FP64";

int num_available_gpus();

void parseRuntimeTarget(const std::filesystem::path &cudaqLibPath,
                        RuntimeTarget &target,
                        const std::string nvqppBuildConfig) {
  simulation_precision precision = simulation_precision::fp32;
  std::optional<std::string> foundPlatformName, foundSimulatorName;
  for (auto &line : cudaq::split(nvqppBuildConfig, '\n')) {
    if (line.find(PLATFORM_LIBRARY) != std::string::npos) {
      cudaq::trim(line);
      auto platformName = cudaq::split(line, '=')[1];
      // Post-process the string
      platformName.erase(
          std::remove(platformName.begin(), platformName.end(), '\"'),
          platformName.end());
      platformName = std::regex_replace(platformName, std::regex("-"), "_");
      foundPlatformName = platformName;
    } else if (line.find(NVQIR_SIMULATION_BACKEND) != std::string::npos &&
               !foundSimulatorName.has_value()) {
      cudaq::trim(line);
      auto simulatorName = cudaq::split(line, '=')[1];
      // Post-process the string
      simulatorName.erase(
          std::remove(simulatorName.begin(), simulatorName.end(), '\"'),
          simulatorName.end());

#if defined(__APPLE__) && defined(__MACH__)
      const std::string libSuffix = "dylib";
#else
      const std::string libSuffix = "so";
#endif
      CUDAQ_INFO("CUDA-Q Library Path is {}.", cudaqLibPath.string());
      const auto libName =
          fmt::format("libnvqir-{}.{}", simulatorName, libSuffix);

      if (std::filesystem::exists(cudaqLibPath / libName)) {
        CUDAQ_INFO("Use {} simulator for target {}", simulatorName,
                   target.name);
        foundSimulatorName =
            std::regex_replace(simulatorName, std::regex("-"), "_");
      } else {
        CUDAQ_INFO("Skip {} simulator for target {} since it is not available",
                   simulatorName, target.name);
      }
    } else if (line.find(IS_FP64_SIMULATION) != std::string::npos) {
      precision = simulation_precision::fp64;
    }
  }
  target.platformName = foundPlatformName.value_or("default");
  target.simulatorName = foundSimulatorName.value_or("");
  target.precision = precision;
}

/// @brief Search the targets folder in the install for available targets.
void findAvailableTargets(
    const std::filesystem::path &targetPath,
    std::unordered_map<std::string, RuntimeTarget> &targets,
    std::unordered_map<std::string, RuntimeTarget> &simulationTargets) {

  // directory_iterator ordering is unspecified, so sort it to make it
  // repeatable and consistent.
  std::vector<std::filesystem::directory_entry> targetEntries;
  for (const auto &entry : std::filesystem::directory_iterator{targetPath})
    targetEntries.push_back(entry);
  std::sort(targetEntries.begin(), targetEntries.end(),
            [](const std::filesystem::directory_entry &a,
               const std::filesystem::directory_entry &b) {
              return a.path().filename() < b.path().filename();
            });

  // Loop over all target files
  for (const auto &configFile : targetEntries) {
    auto path = configFile.path();
    // They must have a .yml suffix
    const std::string configFileExt = ".yml";
    if (path.extension().string() == configFileExt) {
      auto fileName = path.filename().string();
      auto targetName =
          std::regex_replace(fileName, std::regex(configFileExt), "");
      // Open the file and look for the platform, simulator, and description
      std::ifstream inFile(path.string());
      const std::string configFileContent(
          (std::istreambuf_iterator<char>(inFile)),
          std::istreambuf_iterator<char>());
      cudaq::config::TargetConfig config;
      llvm::yaml::Input Input(configFileContent.c_str());
      Input >> config;
      CUDAQ_INFO("Found Target {} with config file {}", targetName, fileName);
      const std::string defaultTargetConfigStr =
          cudaq::config::processRuntimeArgs(config, {});
      RuntimeTarget target;
      target.config = config;
      target.name = targetName;
      target.description = config.Description;
      auto cudaqLibPath = targetPath.parent_path() / "lib";
      parseRuntimeTarget(cudaqLibPath, target, defaultTargetConfigStr);
      CUDAQ_INFO("Found Target: {} -> (sim={}, platform={})", targetName,
                 target.simulatorName, target.platformName);
      // Add the target.
      targets.emplace(targetName, target);

      simulationTargets.emplace(targetName, target);
    }
  }
}

LinkedLibraryHolder::LinkedLibraryHolder() : availablePlatforms{"default"} {
  ScopedTraceWithContext("LinkedLibraryHolder::constructor");
  CUDAQ_INFO("Init infrastructure for pythonic builder.");

  if (!cudaq::__internal__::canModifyTarget())
    return;

  cudaq::__internal__::CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  libSuffix = "dylib";
  cudaq::__internal__::getCUDAQLibraryPath(&data);
#else
  libSuffix = "so";
  dl_iterate_phdr(cudaq::__internal__::getCUDAQLibraryPath, &data);
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
    auto targetPath = cudaqLibPath.parent_path() / "targets";
    findAvailableTargets(targetPath, targets, simulationTargets);
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
    } else if (simulatorLibPaths.count(iter->second.simulatorName)) {
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

bool LinkedLibraryHolder::hasTarget(const std::string &name) {
  auto iter = targets.find(name);
  if (iter == targets.end())
    return false;

  return true;
}

void LinkedLibraryHolder::setTarget(
    const std::string &targetName,
    std::map<std::string, std::string> extraConfig) {
  // Do not set the default target if the disallow
  // flag has been set.
  if (!cudaq::__internal__::canModifyTarget())
    return;

  auto iter = targets.find(targetName);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + targetName + ").");

  auto &target = iter->second;
  if (!target.config.WarningMsg.empty()) {
    fmt::print(fmt::fg(fmt::color::red), "[warning] ");
    // Output the warning message if any
    fmt::print(fmt::fg(fmt::color::blue), "Target {}: {}\n", target.name,
               target.config.WarningMsg);
  }
  const std::string targetConfigStr =
      cudaq::config::processRuntimeArgs(target.config, extraConfig);
  parseRuntimeTarget(cudaqLibPath, target, targetConfigStr);

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
  auto potentialServerHelperPath =
      cudaqLibPath /
      fmt::format("libcudaq-serverhelper-{}.{}", targetName, libSuffix);
  if (std::filesystem::exists(potentialServerHelperPath) &&
      !libHandles.count(potentialServerHelperPath.string())) {
    void *serverHelperHandle = dlopen(
        potentialServerHelperPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (serverHelperHandle)
      libHandles.emplace(potentialServerHelperPath.string(),
                         serverHelperHandle);
  }
  for (auto &[key, value] : extraConfig)
    backendConfigStr += fmt::format(";{};{}", key, value);

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

std::vector<RuntimeTarget> LinkedLibraryHolder::getTargets() const {
  std::vector<RuntimeTarget> ret;
  for (auto &[name, target] : targets)
    ret.emplace_back(target);
  return ret;
}

std::string python::getTransportLayer(LinkedLibraryHolder *holder) {
  if (holder && cudaq::__internal__::canModifyTarget()) {
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
