/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
  virtual const std::string name() const override { return "sunrise"; }
  virtual void injectExtraPayload(const cudaq::RuntimeTarget &target,
                                  cudaq::ServerMessage &msg) override {
    nlohmann::json_pointer<std::string> path("/foo/bar"); // The path to inject
    msg[path] = "test";
  }
};

__attribute__((constructor)) static void registerSunrisePayloadProvider() {
  cudaq::registerExtraPayloadProvider(std::make_unique<SunriseProvider>());
}
