/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausTrajectory.h"

using namespace cudaq;

// A simple trajectory for testing
static KrausTrajectory makeTrajectory(std::size_t id, double prob,
                                      std::size_t errors = 0) {
  std::vector<KrausSelection> selections;
  for (std::size_t i = 0; i < errors; ++i) {
    selections.push_back(KrausSelection(i, {0}, "h", KrausOperatorType{1}));
  }
  return KrausTrajectory(id, selections, prob, 0);
}

CUDAQ_TEST(KrausTrajectoryTest, DefaultConstruction) {
  KrausTrajectory traj;
  EXPECT_EQ(traj.trajectory_id, 0);
  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_EQ(traj.probability, 0.0);
  EXPECT_EQ(traj.num_shots, 0);
  EXPECT_FALSE(traj.measurement_counts.has_value());
}

CUDAQ_TEST(KrausTrajectoryTest, ParameterizedConstruction) {
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", KrausOperatorType{1}),
      KrausSelection(1, {0, 1}, "cx", KrausOperatorType{0})};

  KrausTrajectory traj(42,         // trajectory_id
                       selections, // kraus_selections
                       0.123,      // probability
                       1000        // num_shots
  );

  EXPECT_EQ(traj.trajectory_id, 42);
  EXPECT_EQ(traj.kraus_selections.size(), 2);
  EXPECT_NEAR(traj.probability, 0.123, 1e-9);
  EXPECT_EQ(traj.num_shots, 1000);
}

CUDAQ_TEST(KrausTrajectoryTest, Equality) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorType{1})};

  KrausTrajectory traj1(1, sels, 0.5, 100);
  KrausTrajectory traj2(1, sels, 0.5, 100);
  KrausTrajectory traj3(2, sels, 0.5, 100); // Different ID

  EXPECT_TRUE(traj1 == traj2);
  EXPECT_FALSE(traj1 == traj3);
}

CUDAQ_TEST(KrausTrajectoryTest, WithResults) {
  KrausTrajectory traj(0, {}, 1.0, 100);

  EXPECT_FALSE(traj.measurement_counts.has_value());

  traj.measurement_counts =
      std::map<std::string, std::size_t>{{"00", 60}, {"11", 40}};

  EXPECT_TRUE(traj.measurement_counts.has_value());
  EXPECT_EQ(traj.measurement_counts->size(), 2);
  EXPECT_EQ(traj.measurement_counts->at("00"), 60);
}

CUDAQ_TEST(KrausTrajectoryTest, EmptyTrajectory) {
  // Trajectory with no noise selections (all gates, no errors)
  KrausTrajectory traj(0, {}, 1.0, 1000);

  EXPECT_EQ(traj.trajectory_id, 0);
  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_NEAR(traj.probability, 1.0, 1e-9);
  EXPECT_EQ(traj.num_shots, 1000);
}

CUDAQ_TEST(KrausTrajectoryTest, MultipleErrors) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorType{1}),
      KrausSelection(1, {0}, "x", KrausOperatorType{1}),
      KrausSelection(2, {0, 1}, "cx", KrausOperatorType{5})};

  KrausTrajectory traj(1, sels, 0.001, 10);

  EXPECT_EQ(traj.kraus_selections.size(), 3);
  EXPECT_NEAR(traj.probability, 0.001, 1e-9);
}

CUDAQ_TEST(KrausTrajectoryTest, MoveSemantics) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorType{1})};
  KrausTrajectory original(42, sels, 0.5, 100);

  KrausTrajectory moved = std::move(original);
  EXPECT_EQ(moved.trajectory_id, 42);
  EXPECT_EQ(moved.kraus_selections.size(), 1);
  EXPECT_NEAR(moved.probability, 0.5, 1e-9);
}

CUDAQ_TEST(KrausTrajectoryTest, CompleteScenario) {
  // Simulate a 2-qubit circuit with 2 noise points
  // H(q0) + depolarization, CX(q0,q1) + depolarization2

  // Trajectory 1: Identity errors (no errors)
  std::vector<KrausSelection> sels1 = {
      KrausSelection(0, {0}, "h", KrausOperatorType::IDENTITY),
      KrausSelection(1, {0, 1}, "cx", KrausOperatorType::IDENTITY)};
  KrausTrajectory traj1(0, sels1, 0.85, 850);

  // Trajectory 2: X error on H, no error on CX
  std::vector<KrausSelection> sels2 = {
      KrausSelection(0, {0}, "h", KrausOperatorType{1}),
      KrausSelection(1, {0, 1}, "cx", KrausOperatorType::IDENTITY)};
  KrausTrajectory traj2(1, sels2, 0.10, 100);

  EXPECT_EQ(traj1.kraus_selections.size(), 2);
  EXPECT_EQ(traj2.kraus_selections.size(), 2);

  double total_prob = traj1.probability + traj2.probability;
  EXPECT_NEAR(total_prob, 0.95, 1e-9);
}

CUDAQ_TEST(KrausTrajectoryTest, ConstexprEquality) {
  KrausSelection sel(0, {0}, "h", KrausOperatorType{1});
  std::vector<KrausSelection> sels1 = {sel};
  std::vector<KrausSelection> sels2 = {sel};

  KrausTrajectory traj1(1, sels1, 0.5, 100);
  KrausTrajectory traj2(1, sels2, 0.5, 100);

  EXPECT_TRUE(traj1 == traj2);
}
