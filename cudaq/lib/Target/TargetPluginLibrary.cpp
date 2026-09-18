/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetPluginLibrary.h"
#include "TargetConfigHelper.h"
#include <dlfcn.h>

cudaq::config::TargetPluginLoadResult cudaq::config::loadTargetPluginLibrary(
    const std::filesystem::path *libraryPath) {
  TargetPluginLoadResult result;

  const std::string name =
      libraryPath ? libraryPath->string() : "<linked-in statically>";
  void *handle =
      dlopen(libraryPath ? name.c_str() : nullptr, RTLD_LOCAL | RTLD_NOW);
  if (!handle) {
    const char *dlError = dlerror();
    result.error = "Unable to load target plugin library '" + name +
                   "': " + (dlError ? dlError : "unknown error");
    return result;
  }

  dlerror(); // clear any prior error
  void *symbol = dlsym(handle, kTargetPluginSymbolName);
  if (!symbol) {
    const char *dlError = dlerror();
    result.error = "Target plugin library '" + name +
                   "' does not export the expected symbol '" +
                   kTargetPluginSymbolName +
                   "' - it was likely built against an incompatible CUDA-Q "
                   "version. (" +
                   (dlError ? dlError : "symbol not found") + ")";
    dlclose(handle);
    return result;
  }

  auto entryPoint = reinterpret_cast<TargetPluginEntryPoint>(symbol);
  const TargetConfig *config = entryPoint();
  if (!config) {
    result.error = "Target plugin library '" + name +
                   "' returned a null target configuration.";
    dlclose(handle);
    return result;
  }

  result.ok = true;
  result.config = *config;
  // Deliberately leave the library loaded.
  return result;
}
