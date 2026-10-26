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

bool isValidExpVal(double value) {
  // give us some wiggle room while keep the tests fast
  return value < -1.1 && value > -2.3;
}
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

CUDAQ_TEST(QuantinuumNGTester, checkSampleSync) {

  auto kernel = cudaq::make_kernel();
  auto qubit = kernel.qalloc(2);
  kernel.h(qubit[0]);
  kernel.mz(qubit[0]);

  auto counts = cudaq::sample(100, kernel);
  counts.dump();
  EXPECT_EQ(counts.size(), 2);
}

CUDAQ_TEST(QuantinuumNGTester, checkObserveAsync) {

  auto [kernel, theta] = cudaq::make_kernel<double>();
  auto qubit = kernel.qalloc(2);
  kernel.x(qubit[0]);
  kernel.ry(theta, qubit[1]);
  kernel.x<cudaq::ctrl>(qubit[1], qubit[0]);

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  auto future = cudaq::observe_async(kernel, h, .59);

  auto result = future.get();
  result.dump();

  printf("ENERGY: %lf\n", result.expectation());
  EXPECT_TRUE(isValidExpVal(result.expectation()));
}

CUDAQ_TEST(QuantinuumNGTester, checkControlledRotations) {
  // Small number of shots as NG device mock always
  // runs in shot-shot mode for QIR output.
  constexpr int numShots = 10;
  // rx: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);
    kernel.mz(controls1);
    kernel.mz(controls2);
    kernel.mz(control3);
    kernel.mz(target);
    std::cout << kernel.to_quake() << "\n";

    auto counts = cudaq::sample(numShots, kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>.
    EXPECT_EQ(counts.count("111111"), numShots);
  }

  // rx: 0.0
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.rx<cudaq::ctrl>(0.0, controls1, controls2, control3, target);
    kernel.mz(controls1);
    kernel.mz(controls2);
    kernel.mz(control3);
    kernel.mz(target);
    auto counts = cudaq::sample(numShots, kernel);
    counts.dump();

    // Target qubit should've stayed in |0>
    EXPECT_EQ(counts.count("111110"), numShots);
  }

  // ry: pi
  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);
    kernel.mz(controls1);
    kernel.mz(controls2);
    kernel.mz(control3);
    kernel.mz(target);
    auto counts = cudaq::sample(numShots, kernel);
    counts.dump();

    // Target qubit should've been rotated to |1>
    EXPECT_EQ(counts.count("111111"), numShots);
  }

  // ry: pi / 2
  {
    cudaq::set_random_seed(4);

    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(2);
    auto controls2 = kernel.qalloc(2);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    // All of our controls in the 1-state.
    kernel.x(controls1);
    kernel.x(controls2);
    kernel.x(control3);

    kernel.ry<cudaq::ctrl>(M_PI_2, controls1, controls2, control3, target);
    kernel.mz(controls1);
    kernel.mz(controls2);
    kernel.mz(control3);
    kernel.mz(target);
    auto counts = cudaq::sample(100, kernel);
    counts.dump();

    // Target qubit should have a 50/50 mix between |0> and |1>
    EXPECT_TRUE(counts.count("111111") < 60);
    EXPECT_TRUE(counts.count("111110") > 40);
  }

  {
    auto kernel = cudaq::make_kernel();
    auto controls1 = kernel.qalloc(3);
    auto controls2 = kernel.qalloc(3);
    auto control3 = kernel.qalloc();
    auto target = kernel.qalloc();

    kernel.x(controls1);
    kernel.x(control3);
    // Should do nothing.
    kernel.x<cudaq::ctrl>(controls1, controls2, control3, target);
    kernel.x(controls2);
    // Should rotate `target`.
    kernel.rx<cudaq::ctrl>(M_PI, controls1, controls2, control3, target);
    kernel.mz(controls1);
    kernel.mz(controls2);
    kernel.mz(control3);
    kernel.mz(target);
    auto counts = cudaq::sample(numShots, kernel);
    counts.dump();
    EXPECT_EQ(counts.count("11111111"), numShots);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
