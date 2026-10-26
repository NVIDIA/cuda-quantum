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

namespace {

namespace {
class DummyDecoderConfig : public cudaq::ExtraPayloadProvider {

  std::string m_configStr;

public:
  DummyDecoderConfig(const std::string &configStr) : m_configStr(configStr) {}
  virtual ~DummyDecoderConfig() = default;
  virtual std::string name() const override { return "dummy"; }
  virtual std::string getPayloadType() const override {
    return "gpu_decoder_config";
  }
  virtual std::string
  getExtraPayload(const cudaq::RuntimeTarget &target) override {
    return m_configStr;
  }
};
} // namespace

} // namespace

CUDAQ_TEST(QuantinuumNGTester, checkGpuDecoderConfig) {

  const std::string invalidConfig = "invalid_yaml: {{test: invalid}[]";
  cudaq::registerExtraPayloadProvider(
      std::make_unique<DummyDecoderConfig>(invalidConfig));

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  // This will throw because the decoder config is invalid YAML.
  EXPECT_ANY_THROW(cudaq::sample(100, kernel));
  // Just some dummy config, valid YAML.
  const std::string validConfig = R"(
---
decoders:
  - id:              0
    type:            fancy-decoder
    syndrome_size:   1000
  )";

  cudaq::registerExtraPayloadProvider(
      std::make_unique<DummyDecoderConfig>(validConfig));
  // No longer throws because the decoder config is valid YAML.
  EXPECT_NO_THROW(cudaq::sample(100, kernel));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
