/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DemScope.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/RecordedCircuit.h"
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace nvqir::dem {

namespace {
/// Ensure the NVQIR plugin shared library is loaded before `from_plugin`
/// obtains its address. Loading on demand keeps the analysis engine
/// self-contained.
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
  // Self-load the requested plugin so the analysis engine works in any
  // process that links `cudaq::cudaq-analysis`, regardless of whether the
  // consumer also linked the plugin's target.
  ensurePluginLoaded(plugin_name);

  // The cast to `nvqir::RecordedCircuit` is the contract check: only
  // backends that implement the capability interface can drive DEM
  // analysis.
  std::string name = "dem";
  auto resetRecordedCircuit = [](CircuitSimulator &sim) {
    if (auto *recorder = dynamic_cast<RecordedCircuit *>(&sim))
      recorder->reset();
    else
      throw std::runtime_error(
          "`nvqir::dem::make_scope`: plugin simulator does not implement "
          "`nvqir::RecordedCircuit` and therefore cannot drive DEM "
          "analysis.");
  };
  // Both hooks call `reset()` intentionally:
  // - `on_enter` guarantees a clean simulator regardless of what a prior
  //   call (which may have ended abnormally) left behind.
  // - `on_exit` runs on every exit path including exceptions
  return AnalysisScope::from_plugin(
      std::move(name), std::move(plugin_name),
      {.on_enter = resetRecordedCircuit, .on_exit = resetRecordedCircuit});
}

} // namespace nvqir::dem
