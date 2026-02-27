/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
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
    selections.push_back(KrausSelection(i, {0}, "h", 1, /*is_error=*/true));
  }
  return KrausTrajectory(id, selections, prob, 0);
}

CUDAQ_TEST(KrausTrajectoryTest, DefaultConstruction) {
  KrausTrajectory traj;
  EXPECT_EQ(traj.trajectory_id, 0);
  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_EQ(traj.probability, 0.0);
  EXPECT_EQ(traj.num_shots, 0);
  EXPECT_TRUE(traj.measurement_counts.empty());
}

CUDAQ_TEST(KrausTrajectoryTest, ParameterizedConstruction) {
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", 1, true), KrausSelection(1, {0, 1}, "cx", 0)};

  KrausTrajectory traj(42,         // trajectory_id
                       selections, // kraus_selections
                       0.123,      // probability
                       1000        // num_shots
  );

  EXPECT_EQ(traj.trajectory_id, 42);
  EXPECT_EQ(traj.kraus_selections.size(), 2);
  EXPECT_NEAR(traj.probability, 0.123, PROBABILITY_EPSILON);
  EXPECT_EQ(traj.num_shots, 1000);
}

CUDAQ_TEST(KrausTrajectoryTest, Equality) {
  std::vector<KrausSelection> sels = {KrausSelection(0, {0}, "h", 1, true)};

  KrausTrajectory traj1(1, sels, 0.5, 100);
  KrausTrajectory traj2(1, sels, 0.5, 100);
  KrausTrajectory traj3(2, sels, 0.5, 100); // Different ID

  EXPECT_TRUE(traj1 == traj2);
  EXPECT_FALSE(traj1 == traj3);
}

CUDAQ_TEST(KrausTrajectoryTest, WithResults) {
  KrausTrajectory traj(0, {}, 1.0, 100);

  EXPECT_TRUE(traj.measurement_counts.empty());

  traj.measurement_counts = CountsDictionary{{"00", 60}, {"11", 40}};

  EXPECT_EQ(traj.measurement_counts.size(), 2);
  EXPECT_EQ(traj.measurement_counts.at("00"), 60);
}

CUDAQ_TEST(KrausTrajectoryTest, EmptyTrajectory) {
  // Trajectory with no noise selections (all gates, no errors)
  KrausTrajectory traj(0, {}, 1.0, 1000);

  EXPECT_EQ(traj.trajectory_id, 0);
  EXPECT_TRUE(traj.kraus_selections.empty());
  EXPECT_NEAR(traj.probability, 1.0, PROBABILITY_EPSILON);
  EXPECT_EQ(traj.num_shots, 1000);
}

CUDAQ_TEST(KrausTrajectoryTest, MultipleErrors) {
  std::vector<KrausSelection> sels = {KrausSelection(0, {0}, "h", 1, true),
                                      KrausSelection(1, {0}, "x", 1, true),
                                      KrausSelection(2, {0, 1}, "cx", 5, true)};

  KrausTrajectory traj(1, sels, 0.001, 10);

  EXPECT_EQ(traj.kraus_selections.size(), 3);
  EXPECT_NEAR(traj.probability, 0.001, PROBABILITY_EPSILON);
}

CUDAQ_TEST(KrausTrajectoryTest, MoveSemantics) {
  std::vector<KrausSelection> sels = {KrausSelection(0, {0}, "h", 1, true)};
  KrausTrajectory original(42, sels, 0.5, 100);

  KrausTrajectory moved = std::move(original);
  EXPECT_EQ(moved.trajectory_id, 42);
  EXPECT_EQ(moved.kraus_selections.size(), 1);
  EXPECT_NEAR(moved.probability, 0.5, PROBABILITY_EPSILON);
}

CUDAQ_TEST(KrausTrajectoryTest, CompleteScenario) {
  // Simulate a 2-qubit circuit with 2 noise points
  // H(q0) + depolarization, CX(q0,q1) + depolarization2

  // Trajectory 1: Identity (no errors)
  std::vector<KrausSelection> sels1 = {KrausSelection(0, {0}, "h", 0),
                                       KrausSelection(1, {0, 1}, "cx", 0)};
  KrausTrajectory traj1(0, sels1, 0.85, 850);

  // Trajectory 2: X error on H, no error on CX
  std::vector<KrausSelection> sels2 = {KrausSelection(0, {0}, "h", 1, true),
                                       KrausSelection(1, {0, 1}, "cx", 0)};
  KrausTrajectory traj2(1, sels2, 0.10, 100);

  EXPECT_EQ(traj1.kraus_selections.size(), 2);
  EXPECT_EQ(traj2.kraus_selections.size(), 2);

  double total_prob = traj1.probability + traj2.probability;
  EXPECT_NEAR(total_prob, 0.95, PROBABILITY_EPSILON);
}

CUDAQ_TEST(KrausTrajectoryTest, ConstexprEquality) {
  KrausSelection sel(0, {0}, "h", 1, true);
  std::vector<KrausSelection> sels1 = {sel};
  std::vector<KrausSelection> sels2 = {sel};

  KrausTrajectory traj1(1, sels1, 0.5, 100);
  KrausTrajectory traj2(1, sels2, 0.5, 100);

  EXPECT_TRUE(traj1 == traj2);
}

CUDAQ_TEST(KrausTrajectoryTest, OrderingValidation) {
  // Ordered trajectory
  std::vector<KrausSelection> ordered = {KrausSelection(0, {0}, "h", 1, true),
                                         KrausSelection(1, {0, 1}, "cx", 0),
                                         KrausSelection(2, {1}, "x", 1, true)};
  KrausTrajectory traj_ordered(1, ordered, 0.5, 100);
  EXPECT_TRUE(traj_ordered.isOrdered());

  // Unordered trajectory (circuit_location out of order)
  std::vector<KrausSelection> unordered = {KrausSelection(0, {0}, "h", 1, true),
                                           KrausSelection(2, {1}, "x", 1, true),
                                           KrausSelection(1, {0, 1}, "cx", 0)};
  KrausTrajectory traj_unordered(2, unordered, 0.5, 100);
  EXPECT_FALSE(traj_unordered.isOrdered());

  // Empty trajectory
  KrausTrajectory traj_empty(3, {}, 1.0, 0);
  EXPECT_TRUE(traj_empty.isOrdered());

  // Single element
  std::vector<KrausSelection> single = {KrausSelection(5, {0}, "h", 1, true)};
  KrausTrajectory traj_single(4, single, 0.5, 100);
  EXPECT_TRUE(traj_single.isOrdered());
}

CUDAQ_TEST(KrausTrajectoryTest, BuilderPattern) {
  std::vector<KrausSelection> selections = {
      KrausSelection(0, {0}, "h", 1, true), KrausSelection(1, {0, 1}, "cx", 0)};

  auto traj = KrausTrajectory::builder()
                  .setId(42)
                  .setSelections(selections)
                  .setProbability(0.123)
                  .build();

  EXPECT_EQ(traj.trajectory_id, 42);
  EXPECT_EQ(traj.kraus_selections.size(), 2);
  EXPECT_NEAR(traj.probability, 0.123, PROBABILITY_EPSILON);

  EXPECT_EQ(traj.num_shots, 0);
  EXPECT_TRUE(traj.measurement_counts.empty());

  EXPECT_THROW(
      {
        auto invalid = KrausTrajectory::builder()
                           .setId(1)
                           .setSelections(selections)
                           .setProbability(1.5)
                           .build();
      },
      std::logic_error);

  EXPECT_THROW(
      {
        auto invalid = KrausTrajectory::builder()
                           .setId(1)
                           .setSelections(selections)
                           .setProbability(-0.1)
                           .build();
      },
      std::logic_error);
}

CUDAQ_TEST(KrausTrajectoryTest, CountErrors) {
  std::vector<KrausSelection> no_errors = {KrausSelection(0, {0}, "h", 0),
                                           KrausSelection(1, {0, 1}, "cx", 0)};
  KrausTrajectory traj_no_errors(0, no_errors, 1.0, 0);
  EXPECT_EQ(traj_no_errors.countErrors(), 0);

  std::vector<KrausSelection> single_error = {
      KrausSelection(0, {0}, "h", 1, true), KrausSelection(1, {0, 1}, "cx", 0)};
  KrausTrajectory traj_single_error(1, single_error, 0.1, 0);
  EXPECT_EQ(traj_single_error.countErrors(), 1);

  std::vector<KrausSelection> multiple_errors = {
      KrausSelection(0, {0}, "h", 1, true),
      KrausSelection(1, {0, 1}, "cx", 2, true),
      KrausSelection(2, {1}, "x", 3, true)};
  KrausTrajectory traj_multiple_errors(2, multiple_errors, 0.001, 0);
  EXPECT_EQ(traj_multiple_errors.countErrors(), 3);

  KrausTrajectory traj_empty(3, {}, 1.0, 0);
  EXPECT_EQ(traj_empty.countErrors(), 0);
}
