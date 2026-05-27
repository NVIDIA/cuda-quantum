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
#include <stdexcept>
#include <string>
#include <utility>

namespace nvqir::dem {

namespace {
/// Ensure the NVQIR plugin shared library is loaded before `from_plugin`
/// obtains its address. Loading on demand keeps the analysis engine
/// self-contained.
void ensurePluginLoaded(const std::string &plugin_name) {
  const std::string lib = "libnvqir-" + plugin_name + ".so";
  void *handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    const char *err = dlerror();
    throw std::runtime_error(
        "`nvqir::dem::make_scope`: failed to load NVQIR plugin '" + lib +
        "': " + (err ? err : "unknown dlerror"));
  }
}
} // namespace

AnalysisScope make_scope(std::string plugin_name) {
  // Self-load the requested plugin so the analysis engine works in any
  // process that links `cudaq::cudaq-analysis`, regardless of whether the
  // consumer also linked the plugin's target.
  ensurePluginLoaded(plugin_name);

  // There is no `on_exit` because the simulator is shared across CUDA-Q
  // infrastructure and clearing state on exit could break a follow-up
  // `cudaq::sample` call.
  //
  // The cast to `nvqir::RecordedCircuit` is the contract check: only
  // backends that implement the capability interface can drive DEM
  // analysis.
  std::string name = "dem";
  return AnalysisScope::from_plugin(
      std::move(name), std::move(plugin_name),
      {.on_enter = [](CircuitSimulator &sim) {
        if (auto *recorder = dynamic_cast<RecordedCircuit *>(&sim))
          recorder->reset();
        else
          throw std::runtime_error(
              "`nvqir::dem::make_scope`: plugin simulator does not implement "
              "`nvqir::RecordedCircuit` and therefore cannot drive DEM "
              "analysis.");
      }});
}

} // namespace nvqir::dem
