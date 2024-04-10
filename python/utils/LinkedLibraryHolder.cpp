/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
constexpr static const char IS_FP64_SIMULATION[] =
    "CUDAQ_SIMULATION_SCALAR_FP64";

/// @brief A utility function to check availability of Nvidia GPUs and return
/// their count
int countGPUs() {
  int retCode = std::system("nvidia-smi >/dev/null 2>&1");
  if (0 != retCode) {
    cudaq::info("nvidia-smi: command not found");
    return -1;
  }

  char tmpFile[] = "/tmp/.cmd.capture.XXXXXX";
  int fileDescriptor = mkstemp(tmpFile);
  if (-1 == fileDescriptor) {
    cudaq::info("Failed to create a temporary file to capture output");
    return -1;
  }

  std::string command = "nvidia-smi -L 2>/dev/null | wc -l >> ";
  command.append(tmpFile);
  retCode = std::system(command.c_str());
  if (0 != retCode) {
    cudaq::info("Encountered error while invoking 'nvidia-smi'");
    return -1;
  }

  std::stringstream buffer;
  buffer << std::ifstream(tmpFile).rdbuf();
  close(fileDescriptor);
  unlink(tmpFile);
  return std::stoi(buffer.str());
}

std::size_t RuntimeTarget::num_qpus() {
  auto &platform = cudaq::get_platform();
  return platform.num_qpus();
}

bool RuntimeTarget::is_remote() {
  auto &platform = cudaq::get_platform();
  return platform.is_remote();
}
bool RuntimeTarget::is_emulated() {
  auto &platform = cudaq::get_platform();
  return platform.is_emulated();
}

simulation_precision RuntimeTarget::get_precision() { return precision; }

/// @brief Search the targets folder in the install for available targets.
void findAvailableTargets(
    const std::filesystem::path &targetPath,
    std::unordered_map<std::string, RuntimeTarget> &targets,
    std::unordered_map<std::string, RuntimeTarget> &simulationTargets) {

  // Loop over all target files
  for (const auto &configFile :
       std::filesystem::directory_iterator{targetPath}) {
    auto path = configFile.path();
    // They must have a .config suffix
    if (path.extension().string() == ".config") {
      bool isSimulationTarget = false;
      // Extract the target name from the file name
      simulation_precision precision = simulation_precision::fp32;
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
            isSimulationTarget = true;
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
          } else if (line.find(IS_FP64_SIMULATION) != std::string::npos) {
            precision = simulation_precision::fp64;
          }
        }
      }

      cudaq::info("Found Target: {} -> (sim={}, platform={})", targetName,
                  simulatorName, platformName);
      // Add the target.
      targets.emplace(targetName,
                      RuntimeTarget{targetName, simulatorName, platformName,
                                    description, precision});
      if (isSimulationTarget) {
        cudaq::info("Found Simulation target: {} -> (sim={}, platform={})",
                    targetName, simulatorName, platformName);
        simulationTargets.emplace(
            targetName, RuntimeTarget{targetName, simulatorName, platformName,
                                      description, precision});
        isSimulationTarget = false;
      }
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
  findAvailableTargets(targetPath, targets, simulationTargets);

  cudaq::info("Init: Library Path is {}.", cudaqLibPath.string());

  // We have to ensure that nvqir and cudaq are loaded
  std::vector<std::filesystem::path> libPaths{
      cudaqLibPath / fmt::format("libnvqir.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq.{}", libSuffix)};

  const char *dynlibs_var = std::getenv("CUDAQ_DYNLIBS");
  if (dynlibs_var != nullptr) {
    std::string dynlib;
    std::stringstream ss((std::string(dynlibs_var)));
    while (std::getline(ss, dynlib, ':')) {
      cudaq::info("Init: add dynamic library path {}.", dynlib);
      libPaths.push_back(dynlib);
    }
  }

  // Load all the defaults
  for (auto &p : libPaths)
    libHandles.emplace(p.string(),
                       dlopen(p.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

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

  // Set the default target
  // If environment variable set with a valid value, use it
  // Otherwise, if GPU(s) available and other dependencies are satisfied, set
  // default to 'nvidia', else to 'qpp-cpu'
  defaultTarget = "qpp-cpu";
  if (countGPUs() > 0) {
    // Before setting the defaultTarget to nvidia, make sure the simulator is
    // available.
    const std::string nvidiaTarget = "nvidia";
    auto iter = targets.find(nvidiaTarget);
    if (iter != targets.end()) {
      auto target = iter->second;
      if (std::find(availableSimulators.begin(), availableSimulators.end(),
                    target.simulatorName) != availableSimulators.end())
        defaultTarget = nvidiaTarget;
      else
        cudaq::info(
            "GPU(s) found but cannot select nvidia target because simulator "
            "is not available. Are all dependencies installed?");
    } else {
      cudaq::info("GPU(s) found but cannot select nvidia target because nvidia "
                  "target not found.");
    }
  }
  auto env = std::getenv("CUDAQ_DEFAULT_SIMULATOR");
  if (env) {
    cudaq::info("'CUDAQ_DEFAULT_SIMULATOR' = {}", env);
    auto iter = simulationTargets.find(env);
    if (iter != simulationTargets.end()) {
      cudaq::info("Valid target");
      defaultTarget = iter->second.name;
    }
  }

  // Initialize current target to default, may be overridden by command line
  // argument or set_target() API
  currentTarget = defaultTarget;

  if (disallowTargetModification)
    return;

  // We'll always start off with the default target
  resetTarget();
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

void LinkedLibraryHolder::resetTarget() { setTarget(defaultTarget); }

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
}

std::vector<RuntimeTarget> LinkedLibraryHolder::getTargets() const {
  std::vector<RuntimeTarget> ret;
  for (auto &[name, target] : targets)
    ret.emplace_back(target);
  return ret;
}

} // namespace cudaq
