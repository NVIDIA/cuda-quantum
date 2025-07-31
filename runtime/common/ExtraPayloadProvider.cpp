/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ExtraPayloadProvider.h"

namespace cudaq {

namespace {
std::vector<std::unique_ptr<ExtraPayloadProvider>> extraPayloadProviders;
} // namespace

void registerExtraPayloadProvider(
    std::unique_ptr<ExtraPayloadProvider> provider) {
  extraPayloadProviders.push_back(std::move(provider));
}

const std::vector<std::unique_ptr<ExtraPayloadProvider>> &
getExtraPayloadProviders() {
  return extraPayloadProviders;
}

} // namespace cudaq
                    