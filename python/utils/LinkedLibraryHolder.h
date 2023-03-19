/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "common/Logger.h"
#include <filesystem>
#include <fmt/core.h>
#include <map>
#include <unordered_map>
#include <vector>

namespace nvqir {
class CircuitSimulator;
}

namespace cudaq {

/// @brief The LinkedLibraryHolder provides a mechanism for
/// dynamically loading and storing the required plugin libraries
/// for the CUDA Quantum runtime within the Python runtime.
class LinkedLibraryHolder {
protected:
  // Store the library suffix, .so or .dylib
  std::string libSuffix = "";

  /// @brief The path to the CUDA Quantum libraries
  std::filesystem::path cudaqLibPath;

  /// @brief Map of path strings to dlopen loaded library
  /// handles.
  std::unordered_map<std::string, void *> libHandles;

  /// @brief Map of available simulators
  std::unordered_map<std::string, nvqir::CircuitSimulator *> simulators;

public:
  LinkedLibraryHolder();
  ~LinkedLibraryHolder();

  /// Return true if the simulator with given name is available.
  bool hasQPU(const std::string &name) const;

  /// @brief Return the names of the available qpu backends
  std::vector<std::string> list_qpus() const;

  /// @brief At initialization, set the name of the QPU
  /// and load the correct library
  void setQPU(const std::string &name,
              std::map<std::string, std::string> extraConfig = {});

  /// @brief At initialization, set the name of the
  /// platform and load the correct library.
  void setPlatform(const std::string &name,
                   std::map<std::string, std::string> extraConfig = {});
};
} // namespace cudaq
