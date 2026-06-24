/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExtraPayloadProvider.h"
#include <algorithm>
#include <string>

namespace cudaq {

namespace {
std::vector<std::unique_ptr<ExtraPayloadProvider>> extraPayloadProviders;
} // namespace

void registerExtraPayloadProvider(
    std::unique_ptr<ExtraPayloadProvider> provider) {
  // Check if this has not been registered already
  const auto it = std::find_if(
      extraPayloadProviders.begin(), extraPayloadProviders.end(),
      [&](const auto &entry) { return entry->name() == provider->name(); });
  if (it != extraPayloadProviders.end()) {
    // If so, replace it.
    *it = std::move(provider);
    return;
  }

  extraPayloadProviders.push_back(std::move(provider));
}

const std::vector<std::unique_ptr<ExtraPayloadProvider>> &
getExtraPayloadProviders() {
  return extraPayloadProviders;
}

} // namespace cudaq
