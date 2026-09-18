/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetDatabase.h"

namespace {
// Forces a hard link-time dependency on the ABI marker symbol the linked
// TargetDatabase.gen.cpp was stamped with.
[[maybe_unused]] void (*const abiVersionLinkCheck)() =
    &CUDAQ_TARGET_DB_ABI_SYMBOL_NAME;
} // namespace

const cudaq::config::TargetConfig *
cudaq::config::lookupBuiltinTarget(std::string_view name) {
  for (const auto &[entryName, entryConfig] : detail::builtinTargetTable())
    if (entryName == name)
      return entryConfig;
  return nullptr;
}

const std::vector<
    std::pair<std::string_view, const cudaq::config::TargetConfig *>> &
cudaq::config::listBuiltinTargets() {
  return detail::builtinTargetTable();
}
