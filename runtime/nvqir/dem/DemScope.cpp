/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DemScope.h"
#include "nvqir/CircuitSimulator.h"
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

extern "C" void __quantum__rt__clear_result_maps();

namespace nvqir::dem {

namespace {
/// Ensure the NVQIR plugin shared library is loaded before `from_plugin`
/// obtains its address.
void ensurePluginLoaded(const std::string &plugin_name) {
#ifdef __APPLE__
  constexpr const char *libExt = ".dylib";
#else
  constexpr const char *libExt = ".so";
#endif
  static std::mutex cacheMutex;
  static std::unordered_set<std::string> loadedPlugins;
  {
    std::lock_guard<std::mutex> g(cacheMutex);
    if (loadedPlugins.count(plugin_name))
      return;
  }
  const std::string lib = "libnvqir-" + plugin_name + libExt;
  void *handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    const char *err = dlerror();
    throw std::runtime_error(
        "`nvqir::dem::make_scope`: failed to load NVQIR plugin '" + lib +
        "': " + (err ? err : "unknown dlerror"));
  }
  {
    std::lock_guard<std::mutex> g(cacheMutex);
    loadedPlugins.insert(plugin_name);
  }
}

} // namespace

AnalysisScope make_scope(std::string plugin_name) {
  ensurePluginLoaded(plugin_name);

  // Result handles belong to the complete DEM execution, not to any nested
  // callable it invokes. Reset them at this analysis boundary so AOT and JIT
  // entry points have the same lifecycle and exception paths cannot leak
  // handles into a later analysis.
  auto clearResultMaps = [](CircuitSimulator &) {
    __quantum__rt__clear_result_maps();
  };
  return AnalysisScope::from_plugin(
      "dem", std::move(plugin_name),
      {.on_enter =
           [clearResultMaps](CircuitSimulator &sim) {
             clearResultMaps(sim);
             sim.endExecution();
           },
       .on_exit = clearResultMaps});
}

} // namespace nvqir::dem
