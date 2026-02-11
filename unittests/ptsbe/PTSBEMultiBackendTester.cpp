/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Integration test for the PTSBE generic execution path across state-vector
// simulator backends. Exercises the full pipeline: trace capture, noise
// extraction, trajectory generation, generic execution, and aggregation.
//
// Excluded: DM (density matrix handles noise natively, no trajectories needed).
#if !defined(CUDAQ_BACKEND_DM)

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEExecutionData.h"
#include "cudaq/ptsbe/PTSBESample.h"
#include "cudaq/ptsbe/PTSBESampleResult.h"

using namespace cudaq::ptsbe;

namespace {

auto ghz10Kernel = []() __qpu__ {
  cudaq::qvector q(10);
  h(q[0]);
  for (int i = 0; i < 9; i++)
    x<cudaq::ctrl>(q[i], q[i + 1]);
  mz(q);
};

} // namespace

CUDAQ_TEST(PTSBEMultiBackendTest, GHZ10WithDepolarizationNoise) {
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));
  noise.add_all_qubit_channel("cx", cudaq::depolarization2(0.01));

  sample_options options;
  options.shots = 10000;
  options.noise = noise;
  options.ptsbe.return_execution_data = true;
  options.ptsbe.max_trajectories = 100;

  auto result = sample(options, ghz10Kernel);

  // Basic sampling checks
  EXPECT_GT(result.size(), 0u);
  EXPECT_EQ(result.get_total_shots(), 10000u);

  auto countAll0 = result.count("0000000000");
  auto countAll1 = result.count("1111111111");
  EXPECT_GT(countAll0 + countAll1, 9000u);

  // Execution data consistency checks
  ASSERT_TRUE(result.has_execution_data());
  const auto &data = result.execution_data();

  EXPECT_EQ(data.count_instructions(TraceInstructionType::Gate), 10);
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Noise), 10);
  EXPECT_EQ(data.count_instructions(TraceInstructionType::Measurement), 10);

  EXPECT_GT(data.trajectories.size(), 0u);
  EXPECT_LE(data.trajectories.size(), 100u);

  std::size_t trajectoryShots = 0;
  for (const auto &traj : data.trajectories)
    trajectoryShots += traj.num_shots;
  EXPECT_EQ(trajectoryShots, 10000u);
}

#endif // !defined(CUDAQ_BACKEND_DM)
