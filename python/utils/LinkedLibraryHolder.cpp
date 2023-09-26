/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

// Our hook into configuring the NVQIR backend.
extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *);
}

namespace cudaq {
void setQuantumPlatformInternal(quantum_platform *p);

constexpr static const char PLATFORM_LIBRARY[] = "PLATFORM_LIBRARY=";
constexpr static const char NVQIR_SIMULATION_BACKEND[] =
    "NVQIR_SIMULATION_BACKEND=";
constexpr static const char TARGET_DESCRIPTION[] = "TARGET_DESCRIPTION=";

std::size_t RuntimeTarget::num_qpus() {
  auto &platform = cudaq::get_platform();
  return platform.num_qpus();
}

/// @brief Search the targets folder in the install for available targets.
void findAvailableTargets(
    const std::filesystem::path &targetPath,
    std::unordered_map<std::string, RuntimeTarget> &targets) {

  // Loop over all target files
  for (const auto &configFile :
       std::filesystem::directory_iterator{targetPath}) {
    auto path = configFile.path();
    // They must have a .config suffix
    if (path.extension().string() == ".config") {

      // Extract the target name from the file name
      auto fileName = path.filename().string();
      auto targetName = std::regex_replace(fileName, std::regex(".config"), "");
      std::string platformName = "default", simulatorName = "qpp",
                  description = "", line;
      {
        // Open the file and look for the platform, simulator, and description
        std::ifstream inFile(path.string());
        while (std::getline(inFile, line)) {
          if (line.find(PLATFORM_LIBRARY) != std::string::npos) {
            cudaq::trim(line);
            platformName = cudaq::split(line, '=')[1];
            // Post-process the string
            platformName.erase(
                std::remove(platformName.begin(), platformName.end(), '\"'),
                platformName.end());
            platformName =
                std::regex_replace(platformName, std::regex("-"), "_");

          } else if (line.find(NVQIR_SIMULATION_BACKEND) != std::string::npos) {
            cudaq::trim(line);
            simulatorName = cudaq::split(line, '=')[1];
            // Post-process the string
            simulatorName.erase(
                std::remove(simulatorName.begin(), simulatorName.end(), '\"'),
                simulatorName.end());
            simulatorName =
                std::regex_replace(simulatorName, std::regex("-"), "_");
          } else if (line.find(TARGET_DESCRIPTION) != std::string::npos) {
            cudaq::trim(line);
            description = cudaq::split(line, '=')[1];
            // Post-process the string
            description.erase(
                std::remove(description.begin(), description.end(), '\"'),
                description.end());
          }
        }
      }

      cudaq::info("Found Target: {} -> (sim={}, platform={})", targetName,
                  simulatorName, platformName);
      // Add the target.
      targets.emplace(targetName, RuntimeTarget{targetName, simulatorName,
                                                platformName, description});
    }
  }
}

LinkedLibraryHolder::LinkedLibraryHolder() {
  cudaq::info("Init infrastructure for pythonic builder.");

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
  auto targetPath = cudaqLibPath.parent_path() / "targets";
  findAvailableTargets(targetPath, targets);

  cudaq::info("Init: Library Path is {}.", cudaqLibPath.string());

  // We have to ensure that nvqir and cudaq are loaded
  std::vector<std::filesystem::path> libPaths{
      cudaqLibPath / fmt::format("libnvqir.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq.{}", libSuffix)};

  const char *statevec_dynlibs_var = std::getenv("CUDAQ_DYNLIBS");
  if (statevec_dynlibs_var != nullptr) {
    std::string statevec_dynlib;
    std::stringstream ss((std::string(statevec_dynlibs_var)));
    while (std::getline(ss, statevec_dynlib, ':')) {
      cudaq::info("Init: add custatevec dynamic library path {}.",
                  statevec_dynlib);
      libPaths.push_back(statevec_dynlib);
    }
  }

  // Load all the defaults
  for (auto &p : libPaths)
    libHandles.emplace(p.string(),
                       dlopen(p.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

  // We will always load the RemoteRestQPU plugin in Python.
  auto potentialPath =
      cudaqLibPath / fmt::format("libcudaq-rest-qpu.{}", libSuffix);
  void *restQpuLibHandle =
      dlopen(potentialPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (restQpuLibHandle)
    libHandles.emplace(potentialPath.string(), restQpuLibHandle);

  // Search for all simulators and create / store them
  for (const auto &library :
       std::filesystem::directory_iterator{cudaqLibPath}) {
    auto path = library.path();
    auto fileName = path.filename().string();
    if (fileName.find("nvqir-") != std::string::npos) {

      // Extract and process the simulator name
      auto simName = std::regex_replace(fileName, std::regex("libnvqir-"), "");
      simName = std::regex_replace(simName, std::regex("-"), "_");
      // Remove the suffix from the library
      auto idx = simName.find_last_of(".");
      simName = simName.substr(0, idx);

      // Store the dlopen handles
      auto iter = libHandles.find(path.string());
      bool loadFailed = false;
      if (iter == libHandles.end()) {
        void *simLibHandle =
            dlopen(path.string().c_str(), RTLD_GLOBAL | RTLD_NOW);
        // Add simulator lib if successfully loaded.
        // Note: there could be potential dlopen failures due to missing
        // dependencies.
        if (simLibHandle)
          libHandles.emplace(path.string(), simLibHandle);
        else {
          loadFailed = true;
          // Retrieve the error message
          char *error_msg = dlerror();
          cudaq::info("Failed to load NVQIR backend '{}' from {}. Error: {}",
                      simName, path.string(),
                      (error_msg ? std::string(error_msg) : "unknown."));
        }
      }

      if (!loadFailed) {
        // Load the plugin and get the CircuitSimulator.
        // Skip adding simulator name to the availableSimulators list if failed
        // to load.
        cudaq::info("Found simulator plugin {}.", simName);
        availableSimulators.push_back(simName);
      }

    } else if (fileName.find("cudaq-platform-") != std::string::npos) {
      // store all available platforms.
      // Extract and process the platform name
      auto platformName =
          std::regex_replace(fileName, std::regex("libcudaq-platform-"), "");
      platformName = std::regex_replace(platformName, std::regex("-"), "_");
      // Remove the suffix from the library
      auto idx = platformName.find_last_of(".");
      platformName = platformName.substr(0, idx);

      auto iter = libHandles.find(path.string());
      if (iter == libHandles.end())
        libHandles.emplace(path.string(), dlopen(path.string().c_str(),
                                                 RTLD_GLOBAL | RTLD_NOW));

      // Load the plugin and get the CircuitSimulator.
      availablePlatforms.push_back(platformName);
      cudaq::info("Found platform plugin {}.", platformName);
    }
  }

  targets.emplace("default",
                  RuntimeTarget{"default", "qpp", "default",
                                "Default OpenMP CPU-only simulated QPU."});

  if (disallowTargetModification)
    return;

  // We'll always start off with the default platform and the QPP simulator
  __nvqir__setCircuitSimulator(getSimulator("qpp"));
  setQuantumPlatformInternal(getPlatform("default"));
}

LinkedLibraryHolder::~LinkedLibraryHolder() {
  for (auto &[name, handle] : libHandles) {
    if (handle)
      dlclose(handle);
  }
}

nvqir::CircuitSimulator *
LinkedLibraryHolder::getSimulator(const std::string &simName) {
  auto end = availableSimulators.end();
  auto iter = std::find(availableSimulators.begin(), end, simName);
  if (iter == end)
    throw std::runtime_error("Invalid simulator requested: " + simName);

  return getUniquePluginInstance<nvqir::CircuitSimulator>(
      std::string("getCircuitSimulator_") + simName);
}

quantum_platform *
LinkedLibraryHolder::getPlatform(const std::string &platformName) {
  auto end = availablePlatforms.end();
  auto iter = std::find(availablePlatforms.begin(), end, platformName);
  if (iter == end)
    throw std::runtime_error("Invalid platform requested: " + platformName);

  return getUniquePluginInstance<quantum_platform>(
      std::string("getQuantumPlatform_") + platformName);
}

void LinkedLibraryHolder::resetTarget() {
  // TODO: create config for default target and use setTarget("qpp") here,
  // instead of having this be a code duplication of the logic below.
  __nvqir__setCircuitSimulator(getSimulator("qpp"));
  auto *platform = getPlatform("default");
  platform->setTargetBackend("qpp");
  setQuantumPlatformInternal(platform);
  currentTarget = "default";
}

RuntimeTarget LinkedLibraryHolder::getTarget(const std::string &name) const {
  auto iter = targets.find(name);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + name + ").");

  return iter->second;
}

RuntimeTarget LinkedLibraryHolder::getTarget() const {
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
  if (disallowTargetModification)
    return;

  auto iter = targets.find(targetName);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + targetName + ").");

  auto target = iter->second;

  cudaq::info("Setting target={} (sim={}, platform={})", targetName,
              target.simulatorName, target.platformName);

  __nvqir__setCircuitSimulator(getSimulator(target.simulatorName));
  auto *platform = getPlatform(target.platformName);

  // Pack the config into the backend string name
  std::string backendConfigStr = targetName;
  for (auto &[key, value] : extraConfig)
    backendConfigStr += fmt::format(";{};{}", key, value);

  platform->setTargetBackend(backendConfigStr);
  setQuantumPlatformInternal(platform);
  currentTarget = targetName;
}

std::vector<RuntimeTarget> LinkedLibraryHolder::getTargets() const {
  std::vector<RuntimeTarget> ret;
  for (auto &[name, target] : targets)
    ret.emplace_back(target);
  return ret;
}

} // namespace cudaq
