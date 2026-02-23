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

using namespace cudaq;
using namespace cudaq::ptsbe;

static KrausTrajectory createTrajectory(std::size_t id, double prob,
                                        std::size_t errors = 0) {
  std::vector<KrausSelection> selections;
  for (std::size_t i = 0; i < errors; ++i) {
    selections.push_back(KrausSelection(i, {0}, "h", 1, /*is_error=*/true));
  }
  return KrausTrajectory(id, selections, prob, 0);
}

static KrausTrajectory createTrajectoryWithSelections(
    std::size_t id, std::vector<KrausSelection> selections, double prob) {
  return KrausTrajectory(id, std::move(selections), prob, 0);
}

TEST(TrajectoryDeduplicationTest, HashSameContentEqual) {
  KrausTrajectory a = createTrajectory(0, 0.5, 1);
  KrausTrajectory b = createTrajectory(1, 0.6, 1);
  EXPECT_EQ(hashTrajectoryContent(a), hashTrajectoryContent(b));
}

TEST(TrajectoryDeduplicationTest, HashDifferentContentDifferent) {
  KrausTrajectory a = createTrajectory(0, 0.5, 0);
  KrausTrajectory b = createTrajectory(0, 0.5, 1);
  EXPECT_NE(hashTrajectoryContent(a), hashTrajectoryContent(b));
}

TEST(TrajectoryDeduplicationTest, EmptyInput) {
  std::vector<KrausTrajectory> empty;
  auto result = deduplicateTrajectories(empty);
  EXPECT_TRUE(result.empty());
}

TEST(TrajectoryDeduplicationTest, SingleTrajectory) {
  std::vector<KrausTrajectory> input = {createTrajectory(0, 0.5, 1)};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 1);
  EXPECT_EQ(result[0].kraus_selections.size(), 1);
  EXPECT_NEAR(result[0].probability, 0.5, PROBABILITY_EPSILON);
}

TEST(TrajectoryDeduplicationTest, TwoIdenticalMergeToOne) {
  KrausTrajectory t = createTrajectory(0, 0.5, 1);
  std::vector<KrausTrajectory> input = {t, t};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[0].kraus_selections.size(), 1);
}

TEST(TrajectoryDeduplicationTest, TwoDistinctNoMerge) {
  std::vector<KrausTrajectory> input = {createTrajectory(0, 0.5, 0),
                                        createTrajectory(1, 0.5, 1)};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].multiplicity, 1);
  EXPECT_EQ(result[1].multiplicity, 1);
}

TEST(TrajectoryDeduplicationTest, ThreeIdenticalMergeToOne) {
  KrausTrajectory t = createTrajectory(0, 0.25, 2);
  std::vector<KrausTrajectory> input = {t, t, t};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 3);
  EXPECT_EQ(result[0].kraus_selections.size(), 2);
}

TEST(TrajectoryDeduplicationTest, TwoPairsTwoUnique) {
  KrausTrajectory a = createTrajectory(0, 0.5, 0);
  KrausTrajectory b = createTrajectory(1, 0.5, 1);
  std::vector<KrausTrajectory> input = {a, b, a, b};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[1].multiplicity, 2);
}

TEST(TrajectoryDeduplicationTest, RepresentativeKeepsFirstProbability) {
  KrausTrajectory t1 = createTrajectory(10, 0.4, 1);
  KrausTrajectory t2 = createTrajectory(20, 0.6, 1);
  std::vector<KrausTrajectory> input = {t1, t2};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_NEAR(result[0].probability, 0.4, PROBABILITY_EPSILON);
  EXPECT_EQ(result[0].trajectory_id, 10);
}

TEST(TrajectoryDeduplicationTest, MultiplicitySumPreserved) {
  KrausTrajectory a = createTrajectory(0, 0.5, 0);
  KrausTrajectory b = createTrajectory(1, 0.5, 1);
  std::vector<KrausTrajectory> input = {a, a, a, b, b};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 2);
  std::size_t total_multiplicity = 0;
  for (const auto &trajectory : result)
    total_multiplicity += trajectory.multiplicity;
  EXPECT_EQ(total_multiplicity, 5);
}

TEST(TrajectoryDeduplicationTest, MultiplicityAlwaysAtLeastOne) {
  std::vector<KrausTrajectory> input = {createTrajectory(0, 0.5, 1)};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_GE(result[0].multiplicity, 1);
}

TEST(TrajectoryDeduplicationTest, DifferentOrderDifferentContent) {
  std::vector<KrausSelection> sel1 = {KrausSelection(0, {0}, "h", 0),
                                      KrausSelection(1, {0}, "x", 1, true)};
  std::vector<KrausSelection> sel2 = {KrausSelection(0, {0}, "h", 1, true),
                                      KrausSelection(1, {0}, "x", 0)};
  std::vector<KrausTrajectory> input = {
      createTrajectoryWithSelections(0, sel1, 0.25),
      createTrajectoryWithSelections(1, sel2, 0.25)};
  auto result = deduplicateTrajectories(input);
  EXPECT_EQ(result.size(), 2);
}

TEST(TrajectoryDeduplicationTest, SameContentDifferentIdAndShots) {
  KrausTrajectory t1 = createTrajectory(0, 0.5, 1);
  t1.num_shots = 100;
  KrausTrajectory t2 = createTrajectory(99, 0.5, 1);
  t2.num_shots = 200;
  std::vector<KrausTrajectory> input = {t1, t2};
  auto result = deduplicateTrajectories(input);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(result[0].multiplicity, 2);
  EXPECT_EQ(result[0].trajectory_id, 0);
  EXPECT_EQ(result[0].num_shots, 100);
}
