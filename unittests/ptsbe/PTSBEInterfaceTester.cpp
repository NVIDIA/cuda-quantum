/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/PTSBEInterface.h"

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

struct NonPTSBESimulator {
  void execute(const PTSBatch &) {}
};

struct WrongReturnTypeSimulator {
  cudaq::sample_result sampleWithPTSBE(const PTSBatch &batch) {
    return cudaq::sample_result{};
  }
};

struct WrongParameterSimulator {
  std::vector<cudaq::sample_result> sampleWithPTSBE(int shots) { return {}; }
};

struct NonConstParameterSimulator {
  std::vector<cudaq::sample_result> sampleWithPTSBE(PTSBatch &batch) {
    return {};
  }
};

struct ConstMethodSimulator {
  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const PTSBatch &batch) const {
    return {};
  }
};

static_assert(PTSBECapable<MockPTSBESimulator>);
static_assert(!PTSBECapable<NonPTSBESimulator>);
static_assert(!PTSBECapable<WrongReturnTypeSimulator>);
static_assert(!PTSBECapable<WrongParameterSimulator>);
static_assert(!PTSBECapable<NonConstParameterSimulator>);
static_assert(PTSBECapable<ConstMethodSimulator>);

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

  batch.measure_qubits = {0, 1, 2};

  EXPECT_EQ(batch.trajectories.size(), 5);
  EXPECT_EQ(batch.measure_qubits.size(), 3);
  EXPECT_EQ(batch.trajectories[2].num_shots, 600);
}

/// Test: Trajectory with KrausSelection noise insertions
CUDAQ_TEST(PTSBEInterfaceTest, TrajectoryWithNoise) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 1000;

  // Add noise selections
  traj.kraus_selections.push_back(
      KrausSelection(0, {0}, "h", KrausOperatorType::IDENTITY));
  traj.kraus_selections.push_back(
      KrausSelection(1, {0, 1}, "cx", static_cast<KrausOperatorType>(2)));
  traj.kraus_selections.push_back(
      KrausSelection(2, {1}, "x", static_cast<KrausOperatorType>(1)));

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
  EXPECT_TRUE(batch.measure_qubits.empty());
}

/// Test: Clean trajectory without noise
CUDAQ_TEST(PTSBEInterfaceTest, CleanTrajectory) {
  KrausTrajectory traj;
  traj.trajectory_id = 0;
  traj.num_shots = 500;

  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_EQ(traj.num_shots, 500);
}

/// Test: Runtime dispatch calls sampleWithPTSBE for PTSBECapable simulators
CUDAQ_TEST(PTSBEInterfaceTest, RuntimeDispatchCallsMock) {
  auto testDispatch = []<typename Sim>(Sim &sim, const PTSBatch &batch) {
    if constexpr (PTSBECapable<Sim>) {
      sim.sampleWithPTSBE(batch);
      return true;
    } else {
      return false;
    }
  };

  MockPTSBESimulator ptsbe_sim;
  NonPTSBESimulator non_ptsbe_sim;
  PTSBatch batch;
  batch.measure_qubits = {0, 1};

  EXPECT_TRUE(testDispatch(ptsbe_sim, batch));
  EXPECT_FALSE(testDispatch(non_ptsbe_sim, batch));
  EXPECT_TRUE(ptsbe_sim.sampleWithPTSBE_called);
}

/// Test: Concept correctly rejects wrong signatures
CUDAQ_TEST(PTSBEInterfaceTest, ConceptRejectsWrongSignatures) {
  EXPECT_FALSE(PTSBECapable<WrongReturnTypeSimulator>);
  EXPECT_FALSE(PTSBECapable<WrongParameterSimulator>);
  EXPECT_FALSE(PTSBECapable<NonConstParameterSimulator>);
  EXPECT_TRUE(PTSBECapable<ConstMethodSimulator>);
}
