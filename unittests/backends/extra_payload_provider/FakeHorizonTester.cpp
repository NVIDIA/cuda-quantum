/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/ExtraPayloadProvider.h"
#include "cudaq/algorithm.h"
#include <gtest/gtest.h>

CUDAQ_TEST(ExtraPayloadProviderTester, checkNoProvier) {
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
  virtual std::string name() const override { return "dummy"; }
  virtual std::string getPayloadType() const override { return "test_type"; }
  virtual std::string
  getExtraPayload(const cudaq::RuntimeTarget &target) override {
    return "test";
  }
};

CUDAQ_TEST(ExtraPayloadProviderTester, checkWrongProvider) {
  // Register a wrong provider
  cudaq::registerExtraPayloadProvider(std::make_unique<DummyProvider>());
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
  virtual std::string name() const override { return "sunrise"; }
  virtual std::string getPayloadType() const override { return "test_type"; }
  virtual std::string
  getExtraPayload(const cudaq::RuntimeTarget &target) override {
    return "test";
  }
};

CUDAQ_TEST(ExtraPayloadProviderTester, checkProvider) {
  cudaq::registerExtraPayloadProvider(std::make_unique<SunriseProvider>());
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
