/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "LinkedLibraryHolder.h"
#include "common/PluginUtils.h"
#include "cudaq/platform.h"
#include "nvqir/CircuitSimulator.h"
#include <fstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <regex>
#include <sstream>

namespace py = pybind11;

// Our hook into configuring the NVQIR backend.
extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *);
}

namespace cudaq {

/// @brief Keep an eye out for requests to ignore
/// target modification.
extern bool disallowTargetModification;

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
      if (iter == libHandles.end())
        libHandles.emplace(path.string(), dlopen(path.string().c_str(),
                                                 RTLD_GLOBAL | RTLD_NOW));

      // Load the plugin and get the CircuitSimulator.
      cudaq::info("Found simulator plugin {}.", simName);
      availableSimulators.push_back(simName);

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

  if (cudaq::disallowTargetModification)
    return;

  // We'll always start off with the default platform and the QPP simulator
  __nvqir__setCircuitSimulator(getSimulator("qpp"));
  setQuantumPlatformInternal(getPlatform("default"));
}

LinkedLibraryHolder::~LinkedLibraryHolder() {
  for (auto &[name, handle] : libHandles)
    dlclose(handle);
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
  __nvqir__setCircuitSimulator(getSimulator("qpp"));
  setQuantumPlatformInternal(getPlatform("default"));
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
  if (cudaq::disallowTargetModification)
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

void bindRuntimeTarget(py::module &mod, LinkedLibraryHolder &holder) {

  py::class_<cudaq::RuntimeTarget>(
      mod, "Target",
      "The `cudaq.Target` represents the underlying infrastructure that CUDA "
      "Quantum kernels will execute on. Instances of `cudaq.Target` describe "
      "what simulator they may leverage, the quantum_platform required for "
      "execution, and a description for the target.")
      .def_readonly("name", &cudaq::RuntimeTarget::name,
                    "The name of the `cudaq.Target`.")
      .def_readonly("simulator", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the simulator this `cudaq.Target` leverages. "
                    "This will be empty for physical QPUs.")
      .def_readonly("platform", &cudaq::RuntimeTarget::simulatorName,
                    "The name of the quantum_platform implementation this "
                    "`cudaq.Target` leverages.")
      .def_readonly("description", &cudaq::RuntimeTarget::simulatorName,
                    "A string describing the features for this `cudaq.Target`.")
      .def("num_qpus", &cudaq::RuntimeTarget::num_qpus,
           "Return the number of QPUs available in this `cudaq.Target`.")
      .def(
          "__str__",
          [](cudaq::RuntimeTarget &self) {
            return fmt::format("Target {}\n\tsimulator={}\n\tplatform={}"
                               "\n\tdescription={}\n",
                               self.name, self.simulatorName, self.platformName,
                               self.description);
          },
          "Persist the information in this `cudaq.Target` to a string.");

  mod.def(
      "has_target",
      [&](const std::string &name) { return holder.hasTarget(name); },
      "Return true if the `cudaq.Target` with the given name exists.");
  mod.def(
      "reset_target", [&]() { return holder.resetTarget(); },
      "Reset the current `cudaq.Target` to the default.");
  mod.def(
      "get_target",
      [&](const std::string &name) { return holder.getTarget(name); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_target", [&]() { return holder.getTarget(); },
      "Return the `cudaq.Target` with the given name. Will raise an exception "
      "if the name is not valid.");
  mod.def(
      "get_targets", [&]() { return holder.getTargets(); },
      "Return all available `cudaq.Target` instances on the current system.");
  mod.def(
      "set_target",
      [&](const cudaq::RuntimeTarget &target, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          std::string strValue = "";
          if (py::isinstance<py::bool_>(value))
            strValue = value.cast<py::bool_>() ? "true" : "false";
          else if (py::isinstance<py::str>(value))
            strValue = value.cast<std::string>();
          else
            throw std::runtime_error(
                "QPU kwargs config value must be cast-able to a string.");

          config.emplace(key.cast<std::string>(), strValue);
        }
        holder.setTarget(target.name, config);
      },
      "Set the `cudaq.Target` to be used for CUDA Quantum kernel execution. "
      "Can provide optional, target-specific configuration data via Python "
      "kwargs.");
  mod.def(
      "set_target",
      [&](const std::string &name, py::kwargs extraConfig) {
        std::map<std::string, std::string> config;
        for (auto &[key, value] : extraConfig) {
          std::string strValue = "";
          if (py::isinstance<py::bool_>(value))
            strValue = value.cast<py::bool_>() ? "true" : "false";
          else if (py::isinstance<py::str>(value))
            strValue = value.cast<std::string>();
          else
            throw std::runtime_error(
                "QPU kwargs config value must be cast-able to a string.");

          config.emplace(key.cast<std::string>(), strValue);
        }
        holder.setTarget(name, config);
      },
      "Set the `cudaq.Target` with given name to be used for CUDA Quantum "
      "kernel execution. Can provide optional, target-specific configuration "
      "data via Python kwargs.");
}

} // namespace cudaq
