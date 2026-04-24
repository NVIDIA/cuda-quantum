/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/KrausTrajectory.h"
#include "cudaq/ptsbe/TrajectoryDeduplication.h"
#include <gtest/gtest.h>
#include <vector>

static cudaq::KrausTrajectory createTrajectory(std::size_t id, double prob,
                                               std::size_t errors = 0) {
  std::vector<cudaq::KrausSelection> selections;
  for (std::size_t i = 0; i < errors; ++i) {
    selections.push_back(
        cudaq::KrausSelection(i, {0}, "h", 1, /*is_error=*/true));
  }
  return cudaq::KrausTrajectory(id, selections, prob, 0);
}

static cudaq::KrausTrajectory
createTrajectoryWithSelections(std::size_t id,
                               std::vector<cudaq::KrausSelection> selections,
                               double prob) {
  return cudaq::KrausTrajectory(id, std::move(selections), prob, 0);
}

TEST(TrajectoryDeduplicationTest, HashSameContentEqual) {
  cudaq::KrausTrajectory a = createTrajectory(0, 0.5, 1);
  cudaq::KrausTrajectory b = createTrajectory(1, 0.6, 1);
  EXPECT_EQ(cudaq::ptsbe::hashTrajectoryContent(a),
            cudaq::ptsbe::hashTrajectoryContent(b));
}

TEST(TrajectoryDeduplicationTest, HashDifferentContentDifferent) {
  cudaq::KrausTrajectory a = createTrajectory(0, 0.5, 0);
  cudaq::KrausTrajectory b = createTrajectory(0, 0.5, 1);
  EXPECT_NE(cudaq::ptsbe::hashTrajectoryContent(a),
            cudaq::ptsbe::hashTrajectoryContent(b));
}

TEST(TrajectoryDeduplicationTest, EmptyInput) {
  std::vector<cudaq::KrausTrajectory> empty;
  auto result = cudaq::ptsbe::deduplicateTrajectories(empty);
  EXPECT_TRUE(result.empty());
}

TEST(TrajectoryDeduplicationTest, SingleTrajectory) {
  std::vector<cudaq::KrausTrajectory> input = {createTrajectory(0, 0.5, 1)};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 1);
  EXPECT_EQ(result[0].kraus_selections.size(), 1);
  EXPECT_NEAR(result[0].probability, 0.5, cudaq::PROBABILITY_EPSILON);
}

TEST(TrajectoryDeduplicationTest, TwoIdenticalMergeToOne) {
  cudaq::KrausTrajectory t = createTrajectory(0, 0.5, 1);
  std::vector<cudaq::KrausTrajectory> input = {t, t};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[0].kraus_selections.size(), 1);
}

TEST(TrajectoryDeduplicationTest, TwoDistinctNoMerge) {
  std::vector<cudaq::KrausTrajectory> input = {createTrajectory(0, 0.5, 0),
                                               createTrajectory(1, 0.5, 1)};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].multiplicity, 1);
  EXPECT_EQ(result[1].multiplicity, 1);
}

TEST(TrajectoryDeduplicationTest, ThreeIdenticalMergeToOne) {
  cudaq::KrausTrajectory t = createTrajectory(0, 0.25, 2);
  std::vector<cudaq::KrausTrajectory> input = {t, t, t};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 3);
  EXPECT_EQ(result[0].kraus_selections.size(), 2);
}

TEST(TrajectoryDeduplicationTest, TwoPairsTwoUnique) {
  cudaq::KrausTrajectory a = createTrajectory(0, 0.5, 0);
  cudaq::KrausTrajectory b = createTrajectory(1, 0.5, 1);
  std::vector<cudaq::KrausTrajectory> input = {a, b, a, b};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[1].multiplicity, 2);
}

TEST(TrajectoryDeduplicationTest, RepresentativeKeepsFirstProbability) {
  cudaq::KrausTrajectory t1 = createTrajectory(10, 0.4, 1);
  cudaq::KrausTrajectory t2 = createTrajectory(20, 0.6, 1);
  std::vector<cudaq::KrausTrajectory> input = {t1, t2};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_NEAR(result[0].probability, 0.4, cudaq::PROBABILITY_EPSILON);
  EXPECT_EQ(result[0].trajectory_id, 10);
}

TEST(TrajectoryDeduplicationTest, MultiplicitySumPreserved) {
  cudaq::KrausTrajectory a = createTrajectory(0, 0.5, 0);
  cudaq::KrausTrajectory b = createTrajectory(1, 0.5, 1);
  std::vector<cudaq::KrausTrajectory> input = {a, a, a, b, b};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  std::size_t total_multiplicity = 0;
  for (const auto &trajectory : result)
    total_multiplicity += trajectory.multiplicity;
  EXPECT_EQ(total_multiplicity, 5);
}

TEST(TrajectoryDeduplicationTest, MultiplicityAlwaysAtLeastOne) {
  std::vector<cudaq::KrausTrajectory> input = {createTrajectory(0, 0.5, 1)};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_GE(result[0].multiplicity, 1);
}

TEST(TrajectoryDeduplicationTest, DifferentOrderDifferentContent) {
  std::vector<cudaq::KrausSelection> sel1 = {
      cudaq::KrausSelection(0, {0}, "h", 0),
      cudaq::KrausSelection(1, {0}, "x", 1, true)};
  std::vector<cudaq::KrausSelection> sel2 = {
      cudaq::KrausSelection(0, {0}, "h", 1, true),
      cudaq::KrausSelection(1, {0}, "x", 0)};
  std::vector<cudaq::KrausTrajectory> input = {
      createTrajectoryWithSelections(0, sel1, 0.25),
      createTrajectoryWithSelections(1, sel2, 0.25)};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  EXPECT_EQ(result.size(), 2);
}

TEST(TrajectoryDeduplicationTest, SameContentDifferentIdAndShots) {
  cudaq::KrausTrajectory t1 = createTrajectory(0, 0.5, 1);
  t1.num_shots = 100;
  cudaq::KrausTrajectory t2 = createTrajectory(99, 0.5, 1);
  t2.num_shots = 200;
  std::vector<cudaq::KrausTrajectory> input = {t1, t2};
  auto result = cudaq::ptsbe::deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[0].trajectory_id, 0);
  EXPECT_EQ(result[0].num_shots, 300);
}
