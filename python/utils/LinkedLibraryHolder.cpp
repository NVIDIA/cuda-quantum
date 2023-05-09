/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <fstream>
#include <iostream>
#include <pybind11/pybind11.h>
#include <regex>
#include <sstream>

namespace py = pybind11;

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

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

struct CUDAQLibraryData {
  std::string path;
};

#if defined(__APPLE__) && defined(__MACH__)
static void getCUDAQLibraryPath(CUDAQLibraryData *data) {
  auto nLibs = _dyld_image_count();
  for (uint32_t i = 0; i < nLibs; i++) {
    auto ptr = _dyld_get_image_name(i);
    std::string libName(ptr);
    if (libName.find("cudaq-common") != std::string::npos) {
      auto casted = static_cast<CUDAQLibraryData *>(data);
      casted->path = std::string(ptr);
    }
  }
}
#else
static int getCUDAQLibraryPath(struct dl_phdr_info *info, size_t size,
                               void *data) {
  std::string libraryName(info->dlpi_name);
  if (libraryName.find("cudaq-common") != std::string::npos) {
    auto casted = static_cast<CUDAQLibraryData *>(data);
    casted->path = std::string(info->dlpi_name);
  }
  return 0;
}
#endif

std::size_t RuntimeTarget::num_qpus() {
  auto &platform = cudaq::get_platform();
  return platform.num_qpus();
}

void findAvailableTargets(
    const std::filesystem::path &platformPath,
    std::unordered_map<std::string, RuntimeTarget> &targets) {
  for (const auto &configFile :
       std::filesystem::directory_iterator{platformPath}) {
    auto path = configFile.path();
    auto fileName = path.filename().string();
    if (fileName.find(".config") != std::string::npos) {

      auto targetName = std::regex_replace(fileName, std::regex(".config"), "");
      std::string platformName = "default";
      std::string simulatorName = "qpp";
      std::string description = "";

      std::string line;
      {
        std::ifstream inFile(path.string());
        while (std::getline(inFile, line)) {
          if (line.find(PLATFORM_LIBRARY) != std::string::npos) {
            cudaq::trim(line);
            platformName = cudaq::split(line, '=')[1];
            platformName.erase(
                std::remove(platformName.begin(), platformName.end(), '\"'),
                platformName.end());
            platformName =
                std::regex_replace(platformName, std::regex("-"), "_");

          } else if (line.find(NVQIR_SIMULATION_BACKEND) != std::string::npos) {
            cudaq::trim(line);
            simulatorName = cudaq::split(line, '=')[1];
            simulatorName.erase(
                std::remove(simulatorName.begin(), simulatorName.end(), '\"'),
                simulatorName.end());
            simulatorName =
                std::regex_replace(simulatorName, std::regex("-"), "_");
          } else if (line.find(TARGET_DESCRIPTION) != std::string::npos) {
            cudaq::trim(line);
            description = cudaq::split(line, '=')[1];
            description.erase(
                std::remove(description.begin(), description.end(), '\"'),
                description.end());
          }
        }
      }

      cudaq::info("Found Target: {} -> (sim={}, platform={})", targetName,
                  simulatorName, platformName);
      targets.emplace(targetName, RuntimeTarget{targetName, simulatorName,
                                                platformName, description});
    }
  }
}

LinkedLibraryHolder::LinkedLibraryHolder() {
  cudaq::info("Init infrastructure for pythonic builder.");

  CUDAQLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  libSuffix = "dylib";
  getCUDAQLibraryPath(&data);
#else
  libSuffix = "so";
  dl_iterate_phdr(getCUDAQLibraryPath, &data);
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

  // Start of with just lib nvqir and cudaq, the others are plugins
  // and will be loaded next in setQPU and setPlatform
  std::vector<std::filesystem::path> libPaths{
      cudaqLibPath / fmt::format("libnvqir.{}", libSuffix),
      cudaqLibPath / fmt::format("libcudaq.{}", libSuffix)};

  // Load all the defaults
  for (auto &p : libPaths) {
    libHandles.emplace(p.string(),
                       dlopen(p.string().c_str(), RTLD_GLOBAL | RTLD_NOW));
  }

  // Load all simulators here when we start up.
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

      if (simName == "tensornet" || simName == "cuquantum_mgpu") {
        simulators.emplace(simName, nullptr);
        continue;
      }

      auto iter = libHandles.find(path.string());
      if (iter == libHandles.end())
        libHandles.emplace(path.string(), dlopen(path.string().c_str(),
                                                 RTLD_GLOBAL | RTLD_NOW));

      // Load the plugin and get the CircuitSimulator.
      std::string symbolName = fmt::format("getCircuitSimulator_{}", simName);
      auto *simulator =
          getUniquePluginInstance<nvqir::CircuitSimulator>(symbolName);

      cudaq::info("Found simulator plugin {}.", simName);
      simulators.emplace(simName, simulator);
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
      std::string symbolName =
          fmt::format("getQuantumPlatform_{}", platformName);
      auto *platform =
          getUniquePluginInstance<cudaq::quantum_platform>(symbolName);
      platforms.emplace(platformName, platform);
      cudaq::info("Found platform plugin {}.", platformName);
    }
  }
  __nvqir__setCircuitSimulator(simulators["qpp"]);
  setQuantumPlatformInternal(platforms["default"]);
  targets.emplace("default",
                  RuntimeTarget{"default", "qpp", "default",
                                "Default OpenMP CPU-only simulated QPU."});
}

LinkedLibraryHolder::~LinkedLibraryHolder() {
  for (auto &[name, handle] : libHandles)
    dlclose(handle);
}

void LinkedLibraryHolder::resetTarget() {
  __nvqir__setCircuitSimulator(simulators["qpp"]);
  setQuantumPlatformInternal(platforms["default"]);
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

  auto iter = targets.find(targetName);
  if (iter == targets.end())
    throw std::runtime_error("Invalid target name (" + targetName + ").");

  auto target = iter->second;

  cudaq::info("Setting target={} (sim={}, platform={})", targetName,
              target.simulatorName, target.platformName);

  __nvqir__setCircuitSimulator(simulators[target.simulatorName]);
  auto *platform = platforms[target.platformName];

  // Pack the config into the backend string name
  std::string backendConfigStr = "";
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

// bool LinkedLibraryHolder::hasQPU(const std::string &name) const {
//   std::string mutableName = name;
//   if (name == "cuquantum")
//     mutableName = "custatevec";
//   return simulators.find(mutableName) != simulators.end();
// }

// void LinkedLibraryHolder::setQPU(const std::string &name,
//                                  std::map<std::string, std::string> config) {
//   if (name == "tensornet")
//     throw std::runtime_error(
//         "The tensornet simulator is not available in Python.");

//   std::string mutableName = name;
//   if (name == "cuquantum")
//     mutableName = "custatevec";

//   // Set the simulator if we find it
//   auto iter = simulators.find(mutableName);
//   if (iter != simulators.end()) {
//     if (iter->second) {
//       __nvqir__setCircuitSimulator(iter->second);
//       return;
//     }

//     // If this is one of mpi backends then we need to load it first.
//     // Note we can only load one of these in a single python execution
//     context if (mutableName == "cuquantum_mgpu") {
//       cudaq::info("Requested MPI QPU = {}", mutableName);
//       auto path =
//           cudaqLibPath / fmt::format("libnvqir-{}.{}", mutableName,
//           libSuffix);
//       cudaq::info("Path is {}", path.string());

//       if (!std::filesystem::exists(path))
//         throw std::runtime_error(
//             fmt::format("Invalid path for simulation plugin: {}, {}",
//                         mutableName, path.string()));

//       auto iter = libHandles.find(path.string());
//       if (iter == libHandles.end())
//         libHandles.emplace(path.string(), dlopen(path.string().c_str(),
//                                                  RTLD_GLOBAL | RTLD_NOW));
//       // Load the plugin and get the CircuitSimulator.
//       std::string symbolName =
//           fmt::format("getCircuitSimulator_{}", mutableName);

//       // Load the simulator
//       auto *simulator =
//           getUniquePluginInstance<nvqir::CircuitSimulator>(symbolName);
//       simulators.erase(mutableName);
//       simulators.emplace(mutableName, simulator);
//       __nvqir__setCircuitSimulator(simulator);
//     }
//   }

//   // Check if this name is one of our NAME.config files
//   auto platformPath = cudaqLibPath / ".." / "platforms";
//   if (std::filesystem::exists(platformPath / fmt::format("{}.config", name)))
//   {
//     // Want to setTargetBackend on the platform
//     // May also need to load a plugin library
//     auto potentialPath =
//         cudaqLibPath / fmt::format("libcudaq-rest-qpu.{}", libSuffix);
//     libHandles.emplace(
//         potentialPath.string(),
//         dlopen(potentialPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

//     // Pack the config into the backend string name
//     for (auto &[key, value] : config)
//       mutableName += fmt::format(";{};{}", key, value);

//     cudaq::get_platform().setTargetBackend(mutableName);
//     return;
//   }

//   // Invalid qpu name.
//   throw std::runtime_error("Invalid qpu name: " + name);
// }

// void LinkedLibraryHolder::setPlatform(
//     const std::string &name, std::map<std::string, std::string> config) {

//   std::string mutableName = name;

//   // need to set qpu to cuquantum for mqpu
//   if (name == "mqpu")
//     setQPU("cuquantum");

//   cudaq::info("Setting CUDA Quantum platform to {}.", mutableName);
//   auto potentialPath = cudaqLibPath / fmt::format("libcudaq-platform-{}.{}",
//                                                   mutableName, libSuffix);
//   if (std::filesystem::exists(potentialPath)) {
//     libHandles.emplace(
//         potentialPath.string(),
//         dlopen(potentialPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

//     // Extract the desired quantum_platform subtype and set it on the
//     runtime. std::string symbolName = fmt::format("getQuantumPlatform_{}",
//     mutableName); auto *platform =
//         getUniquePluginInstance<cudaq::quantum_platform>(symbolName);
//     setQuantumPlatformInternal(platform);

//     // Pack the config into the backend string name
//     for (auto &[key, value] : config)
//       mutableName += fmt::format(";{};{}", key, value);

//     platform->setTargetBackend(mutableName);
//     return;
//   }

//   throw std::runtime_error("Invalid platform name: " + name);
// }
} // namespace cudaq
