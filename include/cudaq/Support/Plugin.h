/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "mlir/Pass/Pass.h"
#include <cstdint>
#include <string>

namespace cudaq {

extern "C" {
// Define a C struct that contains the plugin name and
// a callback that will register the pass.
struct PluginLibraryInfo {
  const char *pluginName;
  void (*RegisterCallbacks)();
};
}

/// @brief The Plugin provides an abstraction modeling a
/// plugin library contributing an MLIR Extension (OperationPass, Translation,
/// etc). Plugins are to be loaded from shared libraries on the host system, and
/// they expose a name and callback function that registers the extension with
/// the specific MLIR static registration type.
class Plugin {
private:
  // Constructor, private because clients load the plugin from the static Load
  // method.
  Plugin(const std::string &_fileName, const llvm::sys::DynamicLibrary &lib)
      : fileName(_fileName), library(lib), pluginInfo() {}

  /// @brief The name of the plugin shared library file
  std::string fileName;

  /// @brief The loaded dynamic library
  llvm::sys::DynamicLibrary library;

  /// @brief The plugin information - the name and registration callback.
  PluginLibraryInfo pluginInfo;

public:
  /// @brief Load the plugin at the given shared library filename
  static llvm::Expected<Plugin> Load(const std::string &fileName);

  /// @brief Return the file name this plugin corresponds to
  llvm::StringRef getFilename() const { return fileName; }

  /// @brief Return the plugin name.
  llvm::StringRef getPluginName() const { return pluginInfo.pluginName; }

  /// @brief Register the extension that this plugin provides
  void registerExtensions() const { pluginInfo.RegisterCallbacks(); }
};
} // namespace cudaq

extern "C" ::cudaq::PluginLibraryInfo LLVM_ATTRIBUTE_WEAK cudaqGetPluginInfo();

/// Register a general CUDA Quantum Plugin
#define CUDAQ_REGISTER_MLIR_PLUGIN(NAME, REGISTRATION_FUNCTOR)                 \
  extern "C" LLVM_ATTRIBUTE_WEAK ::cudaq::PluginLibraryInfo                    \
  cudaqGetPluginInfo() {                                                       \
    return cudaq::PluginLibraryInfo{#NAME, REGISTRATION_FUNCTOR};              \
  }

/// Register an MLIR OperationPass
#define CUDAQ_REGISTER_MLIR_PASS(NAME)                                         \
  CUDAQ_REGISTER_MLIR_PLUGIN(NAME, []() { mlir::PassRegistration<NAME>(); })

/// Register an MLIR-to-external Translation
#define CUDAQ_REGISTER_MLIR_TRANSLATION(                                       \
    NAME, DESCRIPTION, TRANSLATION_FUNCTOR, DIALECT_REGISTRY_FUNCTOR)          \
  CUDAQ_REGISTER_MLIR_PLUGIN([]() {                                            \
    TranslateFromMLIRRegistration reg(NAME, DESCRIPTION, TRANSLATION_FUNCTOR,  \
                                      DIALECT_REGISTRY_FUNCTOR);               \
  })
