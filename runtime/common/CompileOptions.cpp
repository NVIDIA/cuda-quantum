/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CompileOptions.h"
#include <mutex>

namespace {
// Constant-initialized so that `nvq++`'s static-initializer setter can run
// before this translation unit's dynamic initialization.
constinit std::mutex g_defaultsMutex;
constinit cudaq::detail::CompileOptionsDefaults g_defaults;
} // namespace

cudaq::detail::CompileOptionsDefaults
cudaq::detail::getCompileOptionsDefaults() {
  std::lock_guard<std::mutex> lock(g_defaultsMutex);
  return g_defaults;
}

void cudaq::detail::setCompileOptionsDefaults(
    const CompileOptionsDefaults &defaults) {
  std::lock_guard<std::mutex> lock(g_defaultsMutex);
  g_defaults = defaults;
}

void cudaq::CompileOptions::populateDefaults() {
  auto defaults = detail::getCompileOptionsDefaults();
  if (!optimizationLevel)
    optimizationLevel = defaults.optimizationLevel;
}
