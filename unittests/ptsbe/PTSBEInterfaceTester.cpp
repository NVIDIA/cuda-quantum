/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEInterface.h"
#include "cudaq/ptsbe/KrausSelection.h"
#include "common/Trace.h"

using namespace cudaq;
using namespace cudaq::ptsbe;

// Compile-time concept validation (if this compiles, concepts work)
namespace {
struct MockPTSBESimulator {
  std::vector<cudaq::ExecutionResult> sampleWithPTSBE(const PTSBatch &batch) {
    return {};
  }
};

struct NonPTSBESimulator {
  void someOtherMethod() {}
};

// These assertions verify concept detection at compile-time
static_assert(PTSBECapable<MockPTSBESimulator>,
              "Mock simulator with sampleWithPTSBE should satisfy concept");
static_assert(!PTSBECapable<NonPTSBESimulator>,
              "Simulator without sampleWithPTSBE should not satisfy concept");
} // namespace

/// Test: PTSBatch compiles and can hold trajectory data
CUDAQ_TEST(PTSBEInterfaceTest, PTSBatchWithTrajectories) {
  PTSBatch batch;

  for (size_t i = 0; i < 5; ++i) {
    PTSBatch::TrajectoryTasks traj;
    traj.trajectory_id = i;
    traj.num_shots = (i + 1) * 200;
    batch.trajectories.push_back(traj);
  }

  batch.measure_qubits = {0, 1, 2};

  EXPECT_EQ(batch.trajectories.size(), 5);
  EXPECT_EQ(batch.measure_qubits.size(), 3);
  EXPECT_EQ(batch.trajectories[2].num_shots, 600);
}

/// Test: Trajectory with KrausSelection noise insertions
CUDAQ_TEST(PTSBEInterfaceTest, TrajectoryWithNoise) {
  PTSBatch::TrajectoryTasks traj;
  traj.trajectory_id = 0;
  traj.num_shots = 1000;

  // Add noise selections
  traj.noise_insertions.push_back(
      KrausSelection(0, {0}, "h", KrausOperatorType::IDENTITY));
  traj.noise_insertions.push_back(
      KrausSelection(1, {0, 1}, "cx", static_cast<KrausOperatorType>(2)));
  traj.noise_insertions.push_back(
      KrausSelection(2, {1}, "x", static_cast<KrausOperatorType>(1)));

  EXPECT_EQ(traj.noise_insertions.size(), 3);
  EXPECT_EQ(traj.noise_insertions[1].qubits.size(), 2);
  EXPECT_EQ(traj.noise_insertions[2].op_name, "x");
}

/// Test: Shot allocation across multiple trajectories
CUDAQ_TEST(PTSBEInterfaceTest, ShotAllocation) {
  PTSBatch batch;

  // Different shot counts per trajectory
  std::vector<size_t> shot_counts = {500, 300, 150, 50};

  for (size_t i = 0; i < shot_counts.size(); ++i) {
    PTSBatch::TrajectoryTasks traj;
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

  PTSBatch::TrajectoryTasks zero_traj;
  zero_traj.trajectory_id = 0;
  zero_traj.num_shots = 0;
  batch.trajectories.push_back(zero_traj);

  PTSBatch::TrajectoryTasks normal_traj;
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
  EXPECT_TRUE(batch.measure_qubits.empty());
}

/// Test: Clean trajectory without noise
CUDAQ_TEST(PTSBEInterfaceTest, CleanTrajectory) {
  PTSBatch::TrajectoryTasks traj;
  traj.trajectory_id = 0;
  traj.num_shots = 500;

  EXPECT_TRUE(traj.noise_insertions.empty());
  EXPECT_EQ(traj.num_shots, 500);
}
