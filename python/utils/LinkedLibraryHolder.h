/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/host_config.h"
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {

class quantum_platform;

/// @brief A RuntimeTarget encapsulates an available
/// backend simulator and quantum_platform for CUDA Quantum
/// kernel execution.
struct RuntimeTarget {
  std::string name;
  std::string simulatorName;
  std::string platformName;
  std::string description;
  simulation_precision precision;

  /// @brief Return the number of QPUs this target exposes.
  std::size_t num_qpus();
  bool is_remote();
  bool is_emulated();
  simulation_precision get_precision();
};

/// @brief The LinkedLibraryHolder provides a mechanism for
/// dynamically loading and storing the required plugin libraries
/// for the CUDA Quantum runtime within the Python runtime.
class LinkedLibraryHolder {
protected:
  // Store the library suffix
  std::string libSuffix = "";

  /// @brief The path to the CUDA Quantum libraries
  std::filesystem::path cudaqLibPath;

  /// @brief Map of path strings to loaded library handles.
  std::unordered_map<std::string, void *> libHandles;

  /// @brief Vector of available simulators
  std::vector<std::string> availableSimulators;

  /// @brief Vector of available platforms
  std::vector<std::string> availablePlatforms;

  /// @brief Map of available targets.
  std::unordered_map<std::string, RuntimeTarget> targets;

  /// @brief Map of simulation targets
  std::unordered_map<std::string, RuntimeTarget> simulationTargets;

  /// @brief Store the name of the default target
  std::string defaultTarget;

  /// @brief Store the name of the current target
  std::string currentTarget;

public:
  LinkedLibraryHolder();
  ~LinkedLibraryHolder();

  /// @brief Return the registered simulator with the given name.
  nvqir::CircuitSimulator *getSimulator(const std::string &name);

  /// @brief Return the registered quantum_platform with the given name.
  quantum_platform *getPlatform(const std::string &name);

  /// @brief Return the available runtime target with given name.
  /// Throws an exception if no target available with that name.
  RuntimeTarget getTarget(const std::string &name) const;

  /// @brief Return the current target.
  RuntimeTarget getTarget() const;

  /// @brief Return all available runtime targets
  std::vector<RuntimeTarget> getTargets() const;

  /// @brief Return true if a target exists with the given name.
  bool hasTarget(const std::string &name);

  /// @brief Set the current target.
  void setTarget(const std::string &targetName,
                 std::map<std::string, std::string> extraConfig = {});

  /// @brief Reset the target back to the default.
  void resetTarget();
};
} // namespace cudaq
