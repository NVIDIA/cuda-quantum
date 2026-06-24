/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExtraPayloadProvider.h"

// Payload provider
class SunriseProvider : public cudaq::ExtraPayloadProvider {
public:
  SunriseProvider() = default;
  virtual ~SunriseProvider() = default;
  virtual std::string name() const override { return "sunrise"; }
  virtual std::string getPayloadType() const override { return "test_type"; }
  virtual std::string
  getExtraPayload(const cudaq::RuntimeTarget &target) override {
    return "test";
  }
};

__attribute__((constructor)) static void registerSunrisePayloadProvider() {
  cudaq::registerExtraPayloadProvider(std::make_unique<SunriseProvider>());
}
