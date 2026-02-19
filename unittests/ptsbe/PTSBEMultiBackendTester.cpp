/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Integration test for the PTSBE generic execution path across simulator
// backends. Exercises the full pipeline: trace capture, noise extraction,
// trajectory generation, generic execution, and aggregation.

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"

using namespace cudaq::ptsbe;

namespace {

auto ghzKernel = []() __qpu__ {
  cudaq::qvector q(5);
  h(q[0]);
  for (int i = 0; i < 4; i++)
    x<cudaq::ctrl>(q[i], q[i + 1]);
  mz(q);
};

struct xMzKernel {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    mz(q);
  }
};

struct twoQubitNoMz {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    x(q[1]);
  }
};

} // namespace

CUDAQ_TEST(PTSBEMultiBackendTest, GHZ3WithDepolarizationNoise) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("cx", cudaq::depolarization2(0.01));

  sample_options options;
  options.shots = 1000;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;
  options.ptsbe.max_trajectories = 20;

  auto result = sample(options, ghzKernel);

  EXPECT_GT(result.size(), 0u);
  EXPECT_EQ(result.get_total_shots(), 1000u);

  auto countAll0 = result.count("00000");
  auto countAll1 = result.count("11111");
  EXPECT_GT(countAll0 + countAll1, 900u);

  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  EXPECT_EQ(data.count_instructions(TraceInstructionType::Gate), 5);
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Noise), 5);
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Measurement), 5);

  EXPECT_GT(data.trajectories.size(), 0u);
  EXPECT_LE(data.trajectories.size(), 20u);

  std::size_t trajectoryShots = 0;
  for (const auto &traj : data.trajectories)
    trajectoryShots += traj.num_shots;
  EXPECT_EQ(trajectoryShots, 1000u);
}

CUDAQ_TEST(PTSBEMultiBackendTest, MzBitFlipFullFlip) {
  cudaq::noise_model noise;
  noise.add_channel("mz", {0}, cudaq::bit_flip_channel(1.0));

  auto result = cudaq::ptsbe::sample(noise, 1000, xMzKernel{});

  EXPECT_GT(result.size(), 0u);
  auto count0 = result.count("0");
  EXPECT_GT(count0, 900u);
}

CUDAQ_TEST(PTSBEMultiBackendTest, ImplicitMzPerQubitNoise) {
  cudaq::noise_model noise;
  noise.add_channel("mz", {0}, cudaq::bit_flip_channel(1.0));
  noise.add_channel("mz", {1}, cudaq::bit_flip_channel(1.0));

  auto result = cudaq::ptsbe::sample(noise, 1000, twoQubitNoMz{});

  EXPECT_EQ(result.get_total_shots(), 1000u);
  EXPECT_GT(result.count("00"), 900u);
}
