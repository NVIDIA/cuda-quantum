/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ExtraPayloadProvider.h"
#include "common/FmtCore.h"
#include "cudaq/algorithm.h"
#include <fstream>
#include <gtest/gtest.h>
#include <regex>

CUDAQ_TEST(ExtraPayloadProviderTester, checkNoProvier) {
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend("horizon;emulate;false"); // disable emulate
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  // Cannot find the provider requested in the config
  EXPECT_THROW(cudaq::sample(kernel), std::runtime_error);
}

class DummyProvider : public cudaq::ExtraPayloadProvider {
public:
  DummyProvider() = default;
  virtual ~DummyProvider() = default;
  virtual const std::string name() const override { return "dummy"; }
  virtual void injectExtraPayload(const cudaq::RuntimeTarget &target,
                                  cudaq::ServerMessage &msg) override {}
};

CUDAQ_TEST(ExtraPayloadProviderTester, checkWrongProvider) {
  // Register a wrong provider
  cudaq::registerExtraPayloadProvider(std::make_unique<DummyProvider>());
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend("horizon;emulate;false"); // disable emulate
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  // Cannot find the provider requested in the config
  EXPECT_THROW(cudaq::sample(kernel), std::runtime_error);
}

// Proper provider requested by the target
class SunriseProvider : public cudaq::ExtraPayloadProvider {
public:
  SunriseProvider() = default;
  virtual ~SunriseProvider() = default;
  virtual const std::string name() const override { return "sunrise"; }
  virtual void injectExtraPayload(const cudaq::RuntimeTarget &target,
                                  cudaq::ServerMessage &msg) override {
    nlohmann::json_pointer<std::string> path(
        target.config.BackendConfig->ExtraPayloadPath);
    msg[path] = "test";
  }
};

CUDAQ_TEST(ExtraPayloadProviderTester, checkProvider) {
  cudaq::registerExtraPayloadProvider(std::make_unique<SunriseProvider>());
  auto &platform = cudaq::get_platform();
  platform.setTargetBackend("horizon;emulate;false"); // disable emulate
  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);
  EXPECT_NO_THROW(cudaq::sample(kernel)); // No throw
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
