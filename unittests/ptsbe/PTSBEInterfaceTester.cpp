/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBESampler.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"
#include <type_traits>

using namespace cudaq;
using namespace cudaq::ptsbe;

namespace {

struct MockPTSBESimulator {
  mutable bool sampleWithPTSBE_called = false;

  std::vector<cudaq::sample_result> sampleWithPTSBE(const PTSBatch &batch) {
    sampleWithPTSBE_called = true;
    return {};
  }
};

struct MockBatchSimulator : BatchSimulator {
  mutable bool sampleWithPTSBE_called = false;

  std::vector<cudaq::sample_result> sampleWithPTSBE(const PTSBatch &batch) {
    sampleWithPTSBE_called = true;
    return {};
  }
};

struct NonPTSBESimulator {
  void execute(const PTSBatch &) {}
};

static_assert(std::is_base_of_v<BatchSimulator, MockBatchSimulator>);
static_assert(!std::is_base_of_v<BatchSimulator, MockPTSBESimulator>);

} // namespace

/// Test: PTSBatch compiles and can hold trajectory data
CUDAQ_TEST(PTSBEInterfaceTest, PTSBatchWithTrajectories) {
  PTSBatch batch;

  for (size_t i = 0; i < 5; ++i) {
    KrausTrajectory traj;
    traj.trajectory_id = i;
    traj.num_shots = (i + 1) * 200;
    batch.trajectories.push_back(traj);
  }

  batch.measureQubits = {0, 1, 2};

  EXPECT_EQ(batch.trajectories.size(), 5);
  EXPECT_EQ(batch.measureQubits.size(), 3);
  EXPECT_EQ(batch.trajectories[2].num_shots, 600);
}

/// Test: Trajectory with KrausSelection noise insertions
CUDAQ_TEST(PTSBEInterfaceTest, TrajectoryWithNoise) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 1000;

  // Add noise selections
  traj.kraus_selections.push_back(KrausSelection(0, {0}, "h", 0));
  traj.kraus_selections.push_back(KrausSelection(1, {0, 1}, "cx", 2, true));
  traj.kraus_selections.push_back(KrausSelection(2, {1}, "x", 1, true));

  EXPECT_EQ(traj.kraus_selections.size(), 3);
  EXPECT_EQ(traj.kraus_selections[1].qubits.size(), 2);
  EXPECT_EQ(traj.kraus_selections[2].op_name, "x");
}

/// Test: Shot allocation across multiple trajectories
CUDAQ_TEST(PTSBEInterfaceTest, ShotAllocation) {
  PTSBatch batch;

  // Different shot counts per trajectory
  std::vector<size_t> shot_counts = {500, 300, 150, 50};

  for (size_t i = 0; i < shot_counts.size(); ++i) {
    KrausTrajectory traj;
    traj.trajectory_id = i;
    traj.num_shots = shot_counts[i];
    batch.trajectories.push_back(traj);
  }

  size_t total = 0;
  for (const auto &t : batch.trajectories)
    total += t.num_shots;

  EXPECT_EQ(total, 1000);
}

/// Test: Zero-shot trajectory (probability thresholding edge case)
CUDAQ_TEST(PTSBEInterfaceTest, ZeroShotTrajectory) {
  PTSBatch batch;

  KrausTrajectory zero_traj;
  zero_traj.trajectory_id = 0;
  zero_traj.num_shots = 0;
  batch.trajectories.push_back(zero_traj);

  KrausTrajectory normal_traj;
  normal_traj.trajectory_id = 1;
  normal_traj.num_shots = 1000;
  batch.trajectories.push_back(normal_traj);

  EXPECT_EQ(batch.trajectories[0].num_shots, 0);
  EXPECT_EQ(batch.trajectories[1].num_shots, 1000);
}

/// Test: Empty batch (validation edge case)
CUDAQ_TEST(PTSBEInterfaceTest, EmptyBatch) {
  PTSBatch batch;

  EXPECT_TRUE(batch.trajectories.empty());
  EXPECT_TRUE(batch.measureQubits.empty());
}

/// Test: Clean trajectory without noise
CUDAQ_TEST(PTSBEInterfaceTest, CleanTrajectory) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 500;

  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_EQ(traj.num_shots, 500);
}

/// Test: Runtime dispatch calls sampleWithPTSBE for BatchSimulator implementers
CUDAQ_TEST(PTSBEInterfaceTest, RuntimeDispatchCallsMock) {
  MockBatchSimulator ptsbe_sim;
  PTSBatch batch;
  batch.measureQubits = {0, 1};

  ptsbe_sim.sampleWithPTSBE(batch);
  EXPECT_TRUE(ptsbe_sim.sampleWithPTSBE_called);

  constexpr bool nonPtsbeIsBatchSimulator =
      std::is_base_of_v<BatchSimulator, NonPTSBESimulator>;
  EXPECT_FALSE(nonPtsbeIsBatchSimulator);
}

/// Test: BatchSimulator inheritance is the dispatch contract
CUDAQ_TEST(PTSBEInterfaceTest, BatchSimulatorDispatchContract) {
  constexpr bool mockBatchIsBatchSimulator =
      std::is_base_of_v<BatchSimulator, MockBatchSimulator>;
  constexpr bool mockPtsbeIsBatchSimulator =
      std::is_base_of_v<BatchSimulator, MockPTSBESimulator>;
  EXPECT_TRUE(mockBatchIsBatchSimulator);
  EXPECT_FALSE(mockPtsbeIsBatchSimulator);
}
