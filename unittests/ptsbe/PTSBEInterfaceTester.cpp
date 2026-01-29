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

// ============================================================================
// MOCK SIMULATORS FOR CONCEPT DETECTION TESTING
// ============================================================================

namespace {

/// @brief Mock simulator with correct sampleWithPTSBE signature
/// @details This simulator satisfies PTSBECapable concept
struct MockPTSBESimulator {
  mutable bool sampleWithPTSBE_called = false;
  mutable PTSBatch last_batch;

  std::vector<cudaq::sample_result> sampleWithPTSBE(const PTSBatch &batch) {
    sampleWithPTSBE_called = true;
    last_batch = batch;
    return {};
  }
};

/// @brief Mock simulator without sampleWithPTSBE method
/// @details This simulator does NOT satisfy PTSBECapable concept
struct NonPTSBESimulator {
  void someOtherMethod() {}
  void execute(const PTSBatch &) {} // Different method, not sampleWithPTSBE
};

/// @brief Mock simulator with wrong return type (T020a edge case)
/// @details Has sampleWithPTSBE but returns wrong type
struct WrongReturnTypeSimulator {
  // Returns single result instead of vector
  cudaq::sample_result sampleWithPTSBE(const PTSBatch &batch) {
    return cudaq::sample_result{};
  }
};

/// @brief Mock simulator with wrong parameter type (T020a edge case)
/// @details Has sampleWithPTSBE but takes wrong parameter
struct WrongParameterSimulator {
  // Takes int instead of PTSBatch
  std::vector<cudaq::sample_result> sampleWithPTSBE(int shots) { return {}; }
};

/// @brief Mock simulator with non-const parameter (T020a edge case)
/// @details Has sampleWithPTSBE but takes non-const reference
struct NonConstParameterSimulator {
  // Takes non-const reference
  std::vector<cudaq::sample_result> sampleWithPTSBE(PTSBatch &batch) {
    return {};
  }
};

/// @brief Mock simulator with const method (should still work)
/// @details Has const sampleWithPTSBE method
struct ConstMethodSimulator {
  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const PTSBatch &batch) const {
    return {};
  }
};

// ============================================================================
// COMPILE-TIME CONCEPT DETECTION (T017)
// ============================================================================

// Primary concept detection tests
static_assert(PTSBECapable<MockPTSBESimulator>,
              "MockPTSBESimulator with correct sampleWithPTSBE must satisfy "
              "PTSBECapable");

static_assert(!PTSBECapable<NonPTSBESimulator>,
              "Simulator without sampleWithPTSBE must NOT satisfy PTSBECapable");

// Edge case: Wrong return type (returns single result, not vector)
static_assert(!PTSBECapable<WrongReturnTypeSimulator>,
              "Simulator with wrong return type must NOT satisfy PTSBECapable");

// Edge case: Wrong parameter type
static_assert(!PTSBECapable<WrongParameterSimulator>,
              "Simulator with wrong parameter type must NOT satisfy "
              "PTSBECapable");

// Edge case: Non-const parameter (concept requires const PTSBatch&)
static_assert(!PTSBECapable<NonConstParameterSimulator>,
              "Simulator with non-const parameter must NOT satisfy "
              "PTSBECapable");

// Edge case: Const method should still work
static_assert(PTSBECapable<ConstMethodSimulator>,
              "Simulator with const sampleWithPTSBE should satisfy "
              "PTSBECapable");

// ============================================================================
// COMPILE-TIME DISPATCH VERIFICATION (T018)
// ============================================================================

/// @brief Helper template to test if constexpr dispatch at compile-time
template <typename Simulator>
constexpr bool dispatchesToOptimizedPath() {
  if constexpr (PTSBECapable<Simulator>) {
    return true; // Would call sampleWithPTSBE
  } else {
    return false; // Would use fallback
  }
}

// Compile-time dispatch verification
static_assert(dispatchesToOptimizedPath<MockPTSBESimulator>(),
              "MockPTSBESimulator must dispatch to optimized path");
static_assert(!dispatchesToOptimizedPath<NonPTSBESimulator>(),
              "NonPTSBESimulator must dispatch to fallback path");
static_assert(!dispatchesToOptimizedPath<WrongReturnTypeSimulator>(),
              "WrongReturnTypeSimulator must dispatch to fallback path");

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

// ============================================================================
// USER STORY 2: CONCEPT DETECTION RUNTIME TESTS (T019-T020)
// ============================================================================

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

/// Test: Concept correctly rejects wrong signatures (T020a edge cases)
CUDAQ_TEST(PTSBEInterfaceTest, ConceptRejectsWrongSignatures) {
  EXPECT_FALSE(PTSBECapable<WrongReturnTypeSimulator>);
  EXPECT_FALSE(PTSBECapable<WrongParameterSimulator>);
  EXPECT_FALSE(PTSBECapable<NonConstParameterSimulator>);
  EXPECT_TRUE(PTSBECapable<ConstMethodSimulator>);
}
