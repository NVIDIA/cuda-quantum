/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "CircuitSimulator.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"
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
      simulators.emplace(simName, simulator);
    }
  }

  // always start with the default, qpp
  setQPU("qpp");
}

LinkedLibraryHolder::~LinkedLibraryHolder() {
  for (auto &[name, handle] : libHandles)
    dlclose(handle);
}

std::vector<std::string> LinkedLibraryHolder::list_qpus() const {
  std::vector<std::string> ret;
  for (auto &[name, ptr] : simulators)
    ret.push_back(name);
  return ret;
}

bool LinkedLibraryHolder::hasQPU(const std::string &name) const {
  std::string mutableName = name;
  if (name == "cuquantum")
    mutableName = "custatevec";
  return simulators.find(mutableName) != simulators.end();
}

void LinkedLibraryHolder::setQPU(const std::string &name,
                                 std::map<std::string, std::string> config) {
  if (name == "tensornet")
    throw std::runtime_error(
        "The tensornet simulator is not available in Python.");

  std::string mutableName = name;
  if (name == "cuquantum")
    mutableName = "custatevec";

  // Set the simulator if we find it
  auto iter = simulators.find(mutableName);
  if (iter != simulators.end()) {
    if (iter->second) {
      __nvqir__setCircuitSimulator(iter->second);
      return;
    }

    // If this is one of mpi backends then we need to load it first.
    // Note we can only load one of these in a single python execution context
    if (mutableName == "cuquantum_mgpu") {
      cudaq::info("Requested MPI QPU = {}", mutableName);
      auto path =
          cudaqLibPath / fmt::format("libnvqir-{}.{}", mutableName, libSuffix);
      cudaq::info("Path is {}", path.string());

      if (!std::filesystem::exists(path))
        throw std::runtime_error(
            fmt::format("Invalid path for simulation plugin: {}, {}",
                        mutableName, path.string()));

      auto iter = libHandles.find(path.string());
      if (iter == libHandles.end())
        libHandles.emplace(path.string(), dlopen(path.string().c_str(),
                                                 RTLD_GLOBAL | RTLD_NOW));
      // Load the plugin and get the CircuitSimulator.
      std::string symbolName =
          fmt::format("getCircuitSimulator_{}", mutableName);

      // Load the simulator
      auto *simulator =
          getUniquePluginInstance<nvqir::CircuitSimulator>(symbolName);
      simulators.erase(mutableName);
      simulators.emplace(mutableName, simulator);
      __nvqir__setCircuitSimulator(simulator);
    }
  }

  // Check if this name is one of our NAME.config files
  auto platformPath = cudaqLibPath / ".." / "platforms";
  if (std::filesystem::exists(platformPath / fmt::format("{}.config", name))) {
    // Want to setTargetBackend on the platform
    // May also need to load a plugin library
    auto potentialPath =
        cudaqLibPath / fmt::format("libcudaq-rest-qpu.{}", libSuffix);
    libHandles.emplace(
        potentialPath.string(),
        dlopen(potentialPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

    // Pack the config into the backend string name
    for (auto &[key, value] : config)
      mutableName += fmt::format(";{};{}", key, value);

    cudaq::get_platform().setTargetBackend(mutableName);
    return;
  }

  // Invalid qpu name.
  throw std::runtime_error("Invalid qpu name: " + name);
}

void LinkedLibraryHolder::setPlatform(
    const std::string &name, std::map<std::string, std::string> config) {

  std::string mutableName = name;
  if (name == "qpud")
    mutableName = "default-qpud";

  // need to set qpu to cuquantum for mqpu
  if (name == "mqpu")
    setQPU("cuquantum");

  cudaq::info("Setting CUDAQ platform to {}.", mutableName);
  auto potentialPath = cudaqLibPath / fmt::format("libcudaq-platform-{}.{}",
                                                  mutableName, libSuffix);
  if (std::filesystem::exists(potentialPath)) {
    libHandles.emplace(
        potentialPath.string(),
        dlopen(potentialPath.string().c_str(), RTLD_GLOBAL | RTLD_NOW));

    // Extract the desired quantum_platform subtype and set it on the runtime.
    std::string symbolName = fmt::format("getQuantumPlatform_{}", mutableName);
    auto *platform =
        getUniquePluginInstance<cudaq::quantum_platform>(symbolName);
    setQuantumPlatformInternal(platform);

    // Pack the config into the backend string name
    for (auto &[key, value] : config)
      mutableName += fmt::format(";{};{}", key, value);

    platform->setTargetBackend(mutableName);
    return;
  }

  throw std::runtime_error("Invalid platform name: " + name);
}
} // namespace cudaq
